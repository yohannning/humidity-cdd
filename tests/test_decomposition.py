import numpy as np
import pandas as pd
import pytest

from cddh import Aggregator, CDDStore, ComfortState, DecompositionAnalysis, GrowthFactorGap, INDICATORS, Zones
from cddh.decomposition import growth_factor, mean_annual_increase

from conftest import GRID


def test_identity_and_symmetric_form():
    g = GrowthFactorGap(A0=500.0, A1=900.0, B0=300.0, B1=650.0)
    assert g.gap == pytest.approx(900 / 500 - 650 / 300)
    assert g.inc + g.base == pytest.approx(g.gap)
    assert g.inc_sym + g.base_sym == pytest.approx(g.gap)
    assert g.max_abs_err < 1e-12
    # equal increments: the gap comes only from the baseline
    h = GrowthFactorGap(400.0, 600.0, 200.0, 400.0)
    assert h.inc == 0 and h.base == pytest.approx(h.gap)
    assert growth_factor(2.0, 3.0) == 1.5 and mean_annual_increase(100, 180, 80) == 1.0


def test_analysis_on_synthetic_store(tmp_path, land, population):
    store, st = CDDStore(tmp_path / "cdd"), ComfortState(22, 60)
    r, c = GRID.rc(land.lat, land.lon)
    for s, (y0, y1) in (("historical", (2001, 2014)), ("ssp245", (2015, 2030))):
        months = pd.date_range(f"{y0}-01-01", f"{y1}-12-01", freq="MS")
        t = months.year.to_numpy()[:, None] - 2000.0
        for k, ind in enumerate(INDICATORS):
            # annual CDD = (100 + 10 k + r) * (1 + 0.02 (k + 1) (year - 2000))
            store.write("toy", s, ind, st, months, np.broadcast_to(
                (100 + 10 * k + r) * (1 + 0.02 * (k + 1) * t) / 12, (len(months), land.n)), land)
    zones = Zones.from_codes(np.ones(GRID.shape, int), ["All"], GRID, {"World": "*"})
    periods = {"base": [2010, 2014], "end": [2026, 2030]}           # as read from a settings file
    an = DecompositionAnalysis(store, Aggregator(population, zones, fill_offgrid=False), periods=periods, log=None)
    df = an.run(["toy"], ["ssp245"], [st])
    assert len(df) == 4 * 2 and set(df.region) == {"All", "World"}
    assert df.max_abs_err.max() < 1e-12
    row = df[(df.pair == "DBT-Standard vs WBT-Standard") & (df.region == "World")].iloc[0]
    w = population.values[r, c]
    w = w * (w > 0)
    mean = lambda x: (x * w).sum() / w.sum()
    A0 = mean((100 + r) * (1 + 0.02 * 12))
    B1 = mean((120 + r) * (1 + 0.06 * 28))
    assert row.A0 == pytest.approx(A0) and row.B1 == pytest.approx(B1)
    s = DecompositionAnalysis.summarize(df)
    assert (s.n_models == 1).all()
    with pytest.raises(ValueError):
        DecompositionAnalysis(store, Aggregator(population, zones, fill_offgrid=True), periods)


def test_periods_are_parsed_and_checked():
    from cddh.decomposition import parse_periods
    assert parse_periods(["base=2010-2019", "end=2090-2099"]) == {"base": (2010, 2019), "end": (2090, 2099)}
    assert parse_periods({"base": [1991, 2020], "mid": [2041, 2070]}) == {"base": (1991, 2020), "mid": (2041, 2070)}
    for bad in (["end=2090-2099"], ["base=2010-2019"], ["base=2019-2010", "end=2090-2099"]):
        with pytest.raises(ValueError):
            parse_periods(bad)
