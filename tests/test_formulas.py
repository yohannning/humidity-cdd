import numpy as np
import pytest

from cddh import DBT_STANDARD, DBT_UKMO, INDICATORS, STATES, WBT_STANDARD, WBT_UKMO, ComfortState, get_indicator
from cddh.formulas import cdd_standard, cdd_ukmo, wbt_stull, wbt_stull_k, wbt_threshold


def test_wbt_threshold_of_main_state():
    assert wbt_threshold(22, 60) == pytest.approx(16.807, abs=5e-4)
    # Stull (2011) check value: 20 C, 50 % -> 13.7 C
    assert wbt_stull(20.0, 50.0) == pytest.approx(13.7, abs=0.05)


def test_wbt_kelvin_and_celsius_agree():
    t = np.linspace(-10, 45, 12)
    rh = np.linspace(10, 95, 12)
    np.testing.assert_allclose(wbt_stull(t, rh), wbt_stull_k(t + 273.15, rh), atol=1e-10)
    assert (wbt_stull(t, rh) <= t + 1e-6).all()


def test_standard_cdd():
    np.testing.assert_array_equal(cdd_standard(np.array([20.0, 22.0, 25.5]), 22.0), [0.0, 0.0, 3.5])


CASES = [  # tmax, tmin, CIBSE TM41 value
    (21.0, 15.0, 0.0),                                  # Tmax <= Tb
    (24.0, 16.0, (24 - 22) / 4),                        # Tmin < Tb < Tmax, Tmean < Tb
    (26.0, 18.0, (26 - 22) / 4),                        # Tmean = Tb
    (30.0, 18.0, (30 - 22) / 2 - (22 - 18) / 4),        # Tmin < Tb < Tmax, Tmean > Tb
    (30.0, 24.0, 27.0 - 22.0),                          # Tmin >= Tb
]


@pytest.mark.parametrize("tmax,tmin,expected", CASES)
def test_ukmo_cases(tmax, tmin, expected):
    assert float(cdd_ukmo(np.array(tmax), np.array(tmin), 22.0)) == pytest.approx(expected)


def test_ukmo_is_continuous_and_below_tmax():
    rng = np.random.default_rng(1)
    tmin = rng.uniform(5, 35, 20000)
    tmax = tmin + rng.uniform(0, 15, 20000)
    v = cdd_ukmo(tmax, tmin, 22.0)
    assert (v >= 0).all() and (v <= np.maximum(tmax - 22.0, 0) + 1e-12).all()
    # TM41 cases meet at Tmean = Tb and Tmin = Tb
    eps = 1e-9
    for tmx, tmn in [(26.0, 18.0), (30.0, 22.0)]:
        assert abs(cdd_ukmo(np.array(tmx + eps), np.array(tmn + eps), 22.0)
                   - cdd_ukmo(np.array(tmx - eps), np.array(tmn - eps), 22.0)) < 1e-6


def test_dtype_is_kept():
    x = np.array([300.0, 290.0], dtype=np.float32)
    assert cdd_standard(x - 273.15, 22.0).dtype == np.float32
    assert wbt_stull_k(x, np.array([50.0, 60.0], dtype=np.float32)).dtype == np.float32


def test_states_and_indicators():
    assert len(STATES) == 8 and ComfortState(22, 60) in STATES
    s = ComfortState.parse("22_60")
    assert (s.label, s.tag, s.csv_name) == ("22_60", "base22C_rh60", "CDD_22C_60p.csv")
    assert DBT_STANDARD.threshold(s) == 22.0
    assert WBT_UKMO.threshold(s) == pytest.approx(16.807, abs=5e-4)
    assert DBT_UKMO.threshold(ComfortState(22, 40)) == DBT_UKMO.threshold(s)
    assert WBT_STANDARD.variables == ("tas", "hurs") and DBT_UKMO.variables == ("tasmax", "tasmin")
    for ind in INDICATORS:
        assert get_indicator(ind.name) is ind and get_indicator(ind.key) is ind
    with pytest.raises(KeyError):
        get_indicator("XYZ")
