"""Heating degree days: formulas (Standard, UKMO/CIBSE TM41), indicators, monthly files and regional tables."""
import numpy as np
import pandas as pd
import pytest

from cddh import (DBT_UKMO, HDD_BASE, HDD_BASES, HDD_INDICATORS, HDD_STANDARD, HDD_UKMO, Aggregator, CDDCalculator,
                  CDDStore, ClimateArchive, ComfortState, Indicator, RegionalCDDTable, get_indicator, hdd_indicators)
from cddh.formulas import cdd_ukmo, hdd_standard, hdd_ukmo

from conftest import MODEL, make_archive

HDD_CASES = [  # tmax, tmin, CIBSE TM41 value for Tb = 18
    (24.0, 19.0, 0.0),                                  # Tmin >= Tb
    (16.0, 10.0, 18.0 - 13.0),                          # Tmax <= Tb
    (20.0, 10.0, (18 - 10) / 2 - (20 - 18) / 4),        # Tmin < Tb < Tmax, Tmean < Tb
    (22.0, 14.0, (18 - 14) / 4),                        # Tmean = Tb
    (26.0, 14.0, (18 - 14) / 4),                        # Tmin < Tb < Tmax, Tmean > Tb
]


def test_standard_hdd():
    np.testing.assert_array_equal(hdd_standard(np.array([20.0, 18.0, 12.5]), 18.0), [0.0, 0.0, 5.5])


@pytest.mark.parametrize("tmax,tmin,expected", HDD_CASES)
def test_ukmo_hdd_cases(tmax, tmin, expected):
    assert float(hdd_ukmo(np.array(tmax), np.array(tmin), 18.0)) == pytest.approx(expected)


def test_ukmo_hdd_mirrors_cdd_and_is_continuous():
    rng = np.random.default_rng(3)
    tmin = rng.uniform(-20, 30, 50000)
    tmax = tmin + rng.uniform(0, 18, 50000)
    v = hdd_ukmo(tmax, tmin, 18.0)
    np.testing.assert_allclose(v, cdd_ukmo(-tmin, -tmax, -18.0), atol=1e-12)
    assert (v >= 0).all() and (v <= np.maximum(18.0 - tmin, 0) + 1e-12).all()
    eps = 1e-9
    for tmx, tmn in [(22.0, 14.0), (18.0, 10.0), (26.0, 18.0)]:   # Tmean = Tb, Tmax = Tb, Tmin = Tb
        assert abs(hdd_ukmo(np.array(tmx + eps), np.array(tmn + eps), 18.0)
                   - hdd_ukmo(np.array(tmx - eps), np.array(tmn - eps), 18.0)) < 1e-6
    assert hdd_ukmo(np.array([10.0], dtype=np.float32), np.array([2.0], dtype=np.float32), 18.0).dtype == np.float32


def test_hdd_indicators_and_bases():
    assert HDD_BASE == ComfortState(18.0) and HDD_BASES == (HDD_BASE,)
    assert (HDD_BASE.label, HDD_BASE.tag, HDD_BASE.csv_name_of("HDD")) == ("18", "base18C", "HDD_18C.csv")
    assert ComfortState.parse("15.5") == ComfortState(15.5)
    assert HDD_STANDARD.name == "HDD-DBT-Standard" and HDD_UKMO.key == "HDD_DBT_UKMO"
    assert HDD_UKMO.threshold(HDD_BASE) == 18.0 and HDD_UKMO.variables == ("tasmax", "tasmin")
    assert HDD_UKMO != DBT_UKMO and DBT_UKMO.name == "DBT-UKMO"           # CDD names are unchanged
    for ind in HDD_INDICATORS:
        assert get_indicator(ind.name) is ind and get_indicator(ind.key) is ind
    assert hdd_indicators(["UKMO"]) == (HDD_UKMO,) and hdd_indicators(["standard", "ukmo"]) == HDD_INDICATORS
    with pytest.raises(KeyError):
        hdd_indicators(["ASHRAE"])
    with pytest.raises(ValueError):
        Indicator("WBT", "UKMO", "HDD")                                    # HDD is dry-bulb only


@pytest.fixture
def hist(tmp_path):
    full = make_archive(tmp_path / "climate", "historical", [(2001, 2002)])
    return ClimateArchive(tmp_path / "climate"), full


def test_monthly_hdd_matches_direct_computation(hist, land):
    arc, full = hist
    bases = [ComfortState(18.0), ComfortState(24.0)]
    months, res = CDDCalculator(land, HDD_INDICATORS, bases).run(arc, MODEL, "historical")
    dates = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    x = {v: full[v].reshape(len(dates), -1)[:, land.mask.ravel()].astype(np.float64) - 273.15
         for v in ("tas", "tasmax", "tasmin")}
    assert len(res) == 4 and len(months) == 24
    for (ind, st), v in res.items():
        daily = (hdd_standard(x["tas"], st.temp_c) if ind.method == "Standard"
                 else hdd_ukmo(x["tasmax"], x["tasmin"], st.temp_c))
        ref = pd.DataFrame(daily).groupby([dates.year, dates.month]).sum().to_numpy()
        np.testing.assert_allclose(v, ref, rtol=2e-5, atol=2e-3)
        assert v.sum() > 0


def test_hdd_store_and_table(hist, land, tmp_path, countries, population):
    arc, _ = hist
    store = CDDStore(tmp_path / "dd")
    calc = CDDCalculator(land, HDD_INDICATORS, HDD_BASES, log=None)
    written = calc.run_to_store(arc, store, MODEL, "historical")
    assert sorted(p.name for p in written) == [
        f"HDD_DBT_Standard_{MODEL}_historical_base18C_monthly.nc",
        f"HDD_DBT_UKMO_{MODEL}_historical_base18C_monthly.nc"]
    assert all(p.parent == tmp_path / "dd" / MODEL / "historical" for p in written)
    months, v, lat, lon = store.read_monthly(MODEL, "historical", HDD_UKMO, HDD_BASE)
    assert v.shape == (24, land.n)
    # a table of the historical run only, both methods
    agg = Aggregator(population, countries.zones("iso3"))
    table = RegionalCDDTable(store, agg, [MODEL], ["historical"], historical=None, indicators=HDD_INDICATORS,
                             log=None)
    df = table.build(HDD_BASE)
    assert list(df.columns) == ["SREGION", "GCM", "SSP", "RCP", "TEMP", "CAL", "YEAR", "HDD"]
    assert set(df.CAL) == {"STANDARD", "UKMO"} and set(df.TEMP) == {"DBT"}
    out = table.write(HDD_BASE, tmp_path / "tables")
    assert out.name == "HDD_18C.csv"
    with pytest.raises(ValueError):
        RegionalCDDTable(store, agg, [MODEL], [], indicators=(HDD_UKMO, DBT_UKMO))
