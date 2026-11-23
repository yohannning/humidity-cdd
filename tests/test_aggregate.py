"""Population weighting, country (ISO3) and region aggregation, regional CDD tables."""
import numpy as np
import pandas as pd
import pytest

from cddh import Aggregator, AnnualCDD, CDDStore, ComfortState, INDICATORS, PopulationGrid, RegionalCDDTable, Zones
from cddh.aggregate import weighted_mean_by_group
from cddh.grid import block_sum
from cddh.tables import COLUMNS

from conftest import GRID


def land_values(land, k=1.0):
    """Value per land cell = k * (10 * row + column)."""
    r, c = GRID.rc(land.lat, land.lon)
    return k * (10.0 * r + c)


def test_weighted_mean_by_group():
    v = np.array([[1.0, 2.0, 3.0, np.nan], [2.0, 4.0, 6.0, 1.0]])
    out = weighted_mean_by_group(v, np.array(["a", "a", "b", "b"]), np.array([1.0, 3.0, 0.0, 5.0]))
    assert set(out) == {"a"}                      # 'b': one weight 0, one non-finite
    np.testing.assert_allclose(out["a"], [(1 + 6) / 4, (2 + 12) / 4])


def test_block_sum_population():
    fine = np.ones((8, 10))
    fine[0, 0] = np.nan
    pop = PopulationGrid.from_fine_array(fine, 2, grid=type(GRID)(10.25, 0.25, 0.5, 4, 5))
    assert pop.values[0, 0] == 3 and pop.total == 79
    np.testing.assert_array_equal(block_sum(np.arange(16).reshape(4, 4), 2), [[10, 18], [42, 50]])


def test_country_means_with_nearest_fill(land, countries, population):
    agg = Aggregator(population, countries.zones("iso3"), fill_offgrid=True)
    v = land_values(land)
    res = agg.means(v, land.lat, land.lon)
    pop = population.values
    # AAA: all cells of columns 0-2 except (3, 0) (CCC) and (1, 1) (no population)
    cells = [(r, c) for r in range(4) for c in range(3) if (r, c) not in [(3, 0), (1, 1)]]
    w = np.array([pop[rc] for rc in cells])
    assert res["AAA"] == pytest.approx(sum(w * [10 * r + c for r, c in cells]) / w.sum())
    # CCC is off the land grid: nearest land cells (2, 0) and (3, 1) are equidistant; the first one in land order wins
    assert res["CCC"] == 20.0
    # BBB: (0, 4) has no country, so only the other cells of columns 3-4
    cells = [(r, c) for r in range(4) for c in (3, 4) if (r, c) != (0, 4)]
    w = np.array([pop[rc] for rc in cells])
    assert res["BBB"] == pytest.approx(sum(w * [10 * r + c for r, c in cells]) / w.sum())


def test_region_is_population_weighted_mean_of_countries(land, countries, population):
    v = np.vstack([land_values(land), land_values(land, 2.0)])
    by_c = Aggregator(population, countries.zones("iso3")).means(v, land.lat, land.lon)
    by_r = Aggregator(population, countries.zones("aim_region", {"World": "*"})).means(v, land.lat, land.lon)
    labels = countries.labels("iso3")
    wsum = {k: population.values[labels == k].sum() for k in ("AAA", "CCC", "BBB")}
    expect = (by_c["AAA"] * wsum["AAA"] + by_c["CCC"] * wsum["CCC"]) / (wsum["AAA"] + wsum["CCC"])
    np.testing.assert_allclose(by_r["R1"], expect)
    np.testing.assert_allclose(by_r["R2"], by_c["BBB"])
    total = sum(wsum.values())
    np.testing.assert_allclose(by_r["World"], sum(by_c[k] * w for k, w in wsum.items()) / total)


def test_land_only_and_min_value(land, countries, population):
    v = land_values(land)
    res = Aggregator(population, countries.zones("iso3"), fill_offgrid=False).means(v, land.lat, land.lon)
    assert "CCC" not in res                                 # no land cell
    res = Aggregator(population, countries.zones("iso3"), min_value=15.0).means(v, land.lat, land.lon)
    assert res["AAA"] > 15 and res["BBB"] > 15
    zero = Aggregator(population, countries.zones("iso3"), min_value=1e6).means(v, land.lat, land.lon)
    assert all((x == 0).all() for x in zero.values())


def test_zones_from_codes():
    codes = np.zeros(GRID.shape, dtype=int)
    codes[:2] = 1
    codes[2:, :3] = 2
    z = Zones.from_codes(codes, ["North", "South"], GRID, {"World": "*", "Both": ["North", "South"]})
    g = z.groups()
    assert g["North"] == {"North"} and g["World"] == {"North", "South"} == g["Both"]
    assert z.labels[3, 4] == ""


def test_grid_mismatch_is_rejected(countries):
    with pytest.raises(ValueError):
        Aggregator(PopulationGrid(np.ones((360, 720))), countries.zones())


def write_store(tmp_path, land, models, scenarios):
    store = CDDStore(tmp_path / "cdd")
    st = ComfortState(22, 60)
    for m in models:
        for s, (y0, y1) in [("historical", (2001, 2014))] + [(s, (2015, 2020)) for s in scenarios]:
            months = pd.date_range(f"{y0}-01-01", f"{y1}-12-01", freq="MS")
            for k, ind in enumerate(INDICATORS):
                vals = np.tile(land_values(land, 1 + k) / 12, (len(months), 1)) + months.year.to_numpy()[:, None] / 12
                store.write(m, s, ind, st, months, vals, land)
    return store, st


def test_regional_table_format(tmp_path, land, countries, population):
    store, st = write_store(tmp_path, land, ["toy-esm1"], ["ssp126", "ssp245"])
    agg = Aggregator(population, countries.zones("aim_region"), min_value=3.0)
    table = RegionalCDDTable(store, agg, ["toy-esm1"], ["ssp126", "ssp245"], log=None)
    path = table.write(st, tmp_path / "out")
    assert path.name == "CDD_22C_60p.csv"
    df = pd.read_csv(path)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2 * 2 * 4 * (14 + 6)                  # regions x scenarios x indicators x years
    assert set(df.GCM) == {"toyesm1"} and set(df.SSP) == {"SSP1", "SSP2"} and set(df.CAL) == {"STANDARD", "UKMO"}
    assert df.YEAR.min() == 2001 and df.YEAR.max() == 2020
    assert df.equals(df.sort_values(["SREGION", "GCM", "RCP", "TEMP", "CAL", "YEAR"]).reset_index(drop=True))
    # values = aggregation of the annual series
    ann = store.read_series("toy-esm1", ["historical", "ssp245"], INDICATORS[2], st)
    ref = agg.means(ann.values, ann.lat, ann.lon)["R2"]
    got = df[(df.SREGION == "R2") & (df.RCP == "ssp245") & (df.TEMP == "WBT") & (df.CAL == "STANDARD")].CDD
    np.testing.assert_allclose(got.to_numpy(), ref, rtol=1e-9)


def test_annual_series_join():
    a = AnnualCDD(np.array([2001, 2002]), np.ones((2, 3)), np.zeros(3), np.zeros(3))
    b = AnnualCDD(np.array([2003]), np.full((1, 3), 4.0), np.zeros(3), np.zeros(3))
    j = a.join(b)
    np.testing.assert_allclose(j.mean(2001, 2003), 2.0)
    with pytest.raises(ValueError):
        j.join(b)
    with pytest.raises(ValueError):
        j.mean(2000, 2003)


def test_nearest_land_ties_and_saved_mapping(tmp_path, land, countries, population):
    from cddh import NearestLand

    # (3, 0) is equidistant from land cells (2, 0) and (3, 1): the smaller land index wins, whatever the k-d tree does
    r, c = GRID.rc(land.lat, land.lon)
    m = NearestLand.compute(r, c, np.array([3]), np.array([0]))
    cand = [int(np.flatnonzero((r == 2) & (c == 0))[0]), int(np.flatnonzero((r == 3) & (c == 1))[0])]
    assert m.land_index[0] == min(cand)
    # a saved mapping is used as is (here: CCC forced to the other candidate)
    rr, cc = np.nonzero((countries.labels() != "") & (population.values > 0))
    full = NearestLand.compute(r, c, rr, cc)
    full.land_index[(rr == 3) & (cc == 0)] = max(cand)
    full.save(tmp_path / "nearest.npz")
    agg = Aggregator(population, countries.zones("iso3"), nearest=NearestLand.load(tmp_path / "nearest.npz"))
    assert agg.means(land_values(land), land.lat, land.lon)["CCC"] == 31.0
    with pytest.warns(UserWarning):     # mapping for other land cells: computed instead
        res = Aggregator(population, countries.zones("iso3"), nearest=NearestLand(rr, cc, full.land_index, r[:-1],
                         c[:-1])).means(land_values(land), land.lat, land.lon)
    assert res["CCC"] == 20.0
