"""Growth factors of CDD indicators and the decomposition of their difference.

GF_X  = X_1 / X_0   (X_0: mean annual CDD in the baseline period, X_1: in a target period)
GF_A - GF_B = (dA - dB) / A0          [increment effect]
            + dB (1/A0 - 1/B0)        [baseline effect]
symmetric form: inc_sym = (dA - dB)(1/A0 + 1/B0)/2,  base_sym = (dA + dB)(1/A0 - 1/B0)/2
X are population-weighted means over the land cells of a zone (fixed population).

Periods are given by the user as {name: (first year, last year)}; the period named 'base' is the baseline and every
other period is a target, e.g. {"base": (2010, 2019), "end": (2090, 2099)}.
"""
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .indicators import DBT_STANDARD, DBT_UKMO, WBT_STANDARD, WBT_UKMO

BASE = "base"
PAIRS = ((DBT_STANDARD, WBT_STANDARD), (DBT_STANDARD, DBT_UKMO), (DBT_STANDARD, WBT_UKMO), (DBT_UKMO, WBT_UKMO))
TOL = 1e-9


@dataclass
class GrowthFactorGap:
    A0: float
    A1: float
    B0: float
    B1: float

    @property
    def dA(self):
        return self.A1 - self.A0

    @property
    def dB(self):
        return self.B1 - self.B0

    @property
    def GF_A(self):
        return self.A1 / self.A0

    @property
    def GF_B(self):
        return self.B1 / self.B0

    @property
    def gap(self):
        return self.GF_A - self.GF_B

    @property
    def inc(self):
        return (self.dA - self.dB) / self.A0

    @property
    def base(self):
        return self.dB * (1 / self.A0 - 1 / self.B0)

    @property
    def inc_sym(self):
        return (self.dA - self.dB) * 0.5 * (1 / self.A0 + 1 / self.B0)

    @property
    def base_sym(self):
        return 0.5 * (self.dA + self.dB) * (1 / self.A0 - 1 / self.B0)

    @property
    def max_abs_err(self):
        return max(abs(self.inc + self.base - self.gap), abs(self.inc_sym + self.base_sym - self.gap))

    def row(self):
        gap, dA = self.gap, self.dA
        return dict(**asdict(self), dA=dA, dB=self.dB, GF_A=self.GF_A, GF_B=self.GF_B, gap=gap,
                    inc=self.inc, base=self.base, inc_sym=self.inc_sym, base_sym=self.base_sym,
                    inc_share=self.inc / gap if gap else np.nan, inc_share_sym=self.inc_sym / gap if gap else np.nan,
                    dB_over_dA=self.dB / dA if dA else np.nan, B0_over_A0=self.B0 / self.A0,
                    max_abs_err=self.max_abs_err)


def growth_factor(x0, x1):
    return x1 / x0


def mean_annual_increase(x0, x1, years):
    """Mean annual increase between two period means `years` apart."""
    return (x1 - x0) / years


def parse_periods(items):
    """['base=2010-2019', 'end=2090-2099'] or {'base': [2010, 2019], ...} -> {'base': (2010, 2019), ...}."""
    if isinstance(items, dict):
        periods = {k: tuple(int(y) for y in v) for k, v in items.items()}
    else:
        periods = {}
        for item in items:
            name, _, span = str(item).partition("=")
            y0, _, y1 = span.partition("-")
            periods[name.strip()] = (int(y0), int(y1 or y0))
    if BASE not in periods or len(periods) < 2:
        raise ValueError(f"periods need '{BASE}' and at least one target period, got {sorted(periods)}")
    for name, (y0, y1) in periods.items():
        if y1 < y0:
            raise ValueError(f"period {name}: {y0}-{y1}")
    return periods


class DecompositionAnalysis:
    """Growth-factor gaps between indicator pairs for every model, scenario, comfort state and zone.

    store      : CDDStore with monthly land-cell CDD (historical + scenarios)
    aggregator : Aggregator with fill_offgrid=False (land cells and their own population)
    periods    : {name: (first, last)} with a 'base' period (see parse_periods)
    """

    def __init__(self, store, aggregator, periods, pairs=PAIRS, historical="historical", log=print):
        if aggregator.fill_offgrid:
            raise ValueError("use an Aggregator with fill_offgrid=False for the decomposition")
        self.store = store
        self.aggregator = aggregator
        self.periods = parse_periods(periods)
        self.pairs = tuple(pairs)
        self.historical = historical
        self.log = log or (lambda *a, **k: None)

    def period_means(self, model, scenario, indicator, state):
        """{zone: {period: population-weighted mean annual CDD}}."""
        scens = ([self.historical] if self.historical else []) + [scenario]
        series = self.store.read_series(model, scens, indicator, state)
        maps = np.vstack([series.mean(*yy) for yy in self.periods.values()])
        res = self.aggregator.means(maps, series.lat, series.lon)
        return {z: dict(zip(self.periods, v)) for z, v in res.items()}

    def decompose(self, model, scenario, state):
        rows, cache = [], {}
        for a, b in self.pairs:
            for ind in (a, b):
                if ind not in cache:
                    cache[ind] = self.period_means(model, scenario, ind, state)
            A, B = cache[a], cache[b]
            for zone in A:
                if zone not in B:
                    continue
                for period in [p for p in self.periods if p != BASE]:
                    g = GrowthFactorGap(A[zone][BASE], A[zone][period], B[zone][BASE], B[zone][period])
                    if g.max_abs_err >= TOL:
                        raise ArithmeticError(f"decomposition does not close: {model} {scenario} {state.label} {zone}")
                    rows.append(dict(reference_state=state.label, pair=f"{a.name} vs {b.name}", model=model,
                                     ssp=scenario, period=period, region=zone, **g.row()))
        return rows

    def run(self, models, scenarios, states):
        t0, rows = time.time(), []
        for state in states:
            for m in models:
                for s in scenarios:
                    rows += self.decompose(m, s, state)
                    self.log(f"  {state.label} {m} {s} ({time.time() - t0:.0f} s)")
        return pd.DataFrame(rows)

    @staticmethod
    def summarize(df):
        """Mean, min and max over models."""
        keys = ["reference_state", "pair", "ssp", "period", "region"]
        vals = ["A0", "B0", "dA", "dB", "GF_A", "GF_B", "gap", "inc", "base", "inc_sym", "base_sym",
                "inc_share", "inc_share_sym", "dB_over_dA", "B0_over_A0"]
        g = df.groupby(keys)
        out = g[vals].agg(["mean", "min", "max"])
        out.columns = [f"{v}_{s}" for v, s in out.columns]
        out["n_models"] = g.model.nunique()
        out["n_gap_positive"] = g.gap.apply(lambda x: int((x > 0).sum()))
        return out.reset_index()
