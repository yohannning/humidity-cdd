"""CDD from daily NetCDF files: compare the chunked, multi-file computation with a direct computation."""
import numpy as np
import pandas as pd
import pytest

from cddh import INDICATORS, STATES, CDDCalculator, CDDStore, ClimateArchive, ComfortState, LandMask
from cddh.formulas import cdd_standard, cdd_ukmo, wbt_stull

from conftest import GRID, MODEL, make_archive

PERIODS = [(2001, 2002), (2003, 2003)]


@pytest.fixture
def archive(tmp_path):
    full = make_archive(tmp_path / "climate", "historical", PERIODS)
    return ClimateArchive(tmp_path / "climate"), full


def direct_monthly(full, land, indicator, state, dates):
    """Reference in float64, cell by cell from the full daily arrays."""
    x = {v: full[v].reshape(len(dates), -1)[:, land.mask.ravel()].astype(np.float64) for v in full}
    th = indicator.threshold(state)
    if indicator.temperature == "DBT":
        t = {v: x[v] - 273.15 for v in ("tas", "tasmax", "tasmin")}
    else:
        t = {v: wbt_stull(x[v] - 273.15, x["hurs"]) for v in ("tas", "tasmax", "tasmin")}
    daily = cdd_standard(t["tas"], th) if indicator.method == "Standard" else cdd_ukmo(t["tasmax"], t["tasmin"], th)
    return pd.DataFrame(daily).groupby([dates.year, dates.month]).sum().to_numpy()


def test_archive_discovery(archive):
    arc, _ = archive
    assert arc.periods(MODEL, "historical") == PERIODS
    assert arc.periods(MODEL, "historical", years=(2003, 2010)) == [(2003, 2003)]
    lat, lon = arc.coords(MODEL, "historical")
    np.testing.assert_allclose(lat, GRID.lat)
    with pytest.raises(FileNotFoundError):
        arc.files(MODEL, "ssp585", "tas")


def test_monthly_cdd_matches_direct_computation(archive, land):
    arc, full = archive
    months, res = CDDCalculator(land).run(arc, MODEL, "historical")
    dates = pd.date_range("2001-01-01", "2003-12-31", freq="D")
    assert len(months) == 36 and months[0] == pd.Timestamp(2001, 1, 1) and months[-1] == pd.Timestamp(2003, 12, 1)
    assert len(res) == len(INDICATORS) * len(STATES)
    for (ind, st), v in res.items():
        assert v.shape == (36, land.n) and v.dtype == np.float32
        np.testing.assert_allclose(v, direct_monthly(full, land, ind, st, dates), rtol=2e-5, atol=2e-3)
    # the synthetic climate is warm enough for every indicator to have cooling degree days
    assert all(v.sum() > 0 for v in res.values())


def test_indicator_ordering_and_rh(archive, land):
    arc, _ = archive
    _, res = CDDCalculator(land).run(arc, MODEL, "historical", years=(2003, 2003))
    by = {(i.name, s.label): v.sum() for (i, s), v in res.items()}
    # DBT does not depend on RH; a higher base temperature gives fewer CDD
    assert by[("DBT-Standard", "22_40")] == by[("DBT-Standard", "22_60")]
    assert by[("WBT-Standard", "22_40")] != by[("WBT-Standard", "22_60")]
    for name in ("DBT-Standard", "DBT-UKMO", "WBT-Standard", "WBT-UKMO"):
        assert by[(name, "20_60")] > by[(name, "22_60")] > by[(name, "24_60")] > by[(name, "26_60")]


def test_result_does_not_depend_on_selection_of_jobs(archive, land):
    arc, _ = archive
    _, all_jobs = CDDCalculator(land).run(arc, MODEL, "historical", years=(2001, 2001))
    ind, st = INDICATORS[3], ComfortState(24, 40)
    _, one = CDDCalculator(land, [ind], [st]).run(arc, MODEL, "historical", years=(2001, 2001))
    np.testing.assert_array_equal(one[(ind, st)], all_jobs[(ind, st)])


def test_store_roundtrip_and_skip(archive, land, tmp_path):
    arc, _ = archive
    store = CDDStore(tmp_path / "cdd")
    calc = CDDCalculator(land, INDICATORS[:2], [ComfortState(22, 60)], log=None)
    written = calc.run_to_store(arc, store, MODEL, "historical")
    assert len(written) == 2 and all(p.exists() for p in written)
    assert calc.run_to_store(arc, store, MODEL, "historical") == []      # existing files are kept
    months, res = calc.run(arc, MODEL, "historical")
    ann = store.read_annual(MODEL, "historical", INDICATORS[0], ComfortState(22, 60))
    assert list(ann.years) == [2001, 2002, 2003]
    np.testing.assert_allclose(ann.values[1], res[(INDICATORS[0], ComfortState(22, 60))][12:24].sum(0), rtol=1e-6)
    np.testing.assert_allclose(ann.lat, land.lat)
    back = LandMask.from_cells(GRID.lat, GRID.lon, ann.lat, ann.lon)
    np.testing.assert_array_equal(back.mask, land.mask)
    land.save(tmp_path / "land.npz")
    again = LandMask.load(tmp_path / "land.npz", GRID.lat, GRID.lon)
    np.testing.assert_array_equal(again.mask, land.mask)
    np.testing.assert_array_equal(again.iso, land.iso)


def test_land_mask_from_shapefile(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    layer = gpd.GeoDataFrame({"iso_a3": ["AAA", "-99"]}, geometry=[box(0, 9, 1.5, 10.5), box(1.5, 9, 2.5, 10.5)],
                             crs=4326)
    layer.to_file(tmp_path / "countries.geojson", driver="GeoJSON")
    m = LandMask.from_shapefile(tmp_path / "countries.geojson", GRID.lat, GRID.lon)
    assert m.n == 3 * 3 and set(m.iso) == {"AAA"}         # rows 9.25-10.25, columns 0.25-1.25; '-99' ignored


def test_explicit_jobs(archive, land):
    arc, _ = archive
    jobs = [(INDICATORS[1], ComfortState(22, 60)), (INDICATORS[3], ComfortState(26, 40))]
    calc = CDDCalculator(land, jobs=jobs, log=None)
    assert calc.jobs == jobs and set(calc.variables) == {"tasmax", "tasmin", "hurs"}
    _, res = calc.run(arc, MODEL, "historical", years=(2002, 2002))
    _, full = CDDCalculator(land, log=None).run(arc, MODEL, "historical", years=(2002, 2002))
    assert set(res) == set(jobs)
    for j in jobs:
        np.testing.assert_array_equal(res[j], full[j])
