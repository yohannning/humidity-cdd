"""Regional degree-day tables (CDD or HDD) in the AIM/Hub degree-day input format.

One CSV per comfort state (CDD_<T>C_<RH>p.csv) or HDD base temperature (HDD_<T>C.csv), long format:
    SREGION, GCM, SSP, RCP, TEMP, CAL, YEAR, CDD   (last column HDD for heating tables)
SREGION = zone label (ISO3 country or AIM/Hub region), GCM = model name without separators (as used in AIM/Hub),
SSP = 'SSP2' for 'ssp245', RCP = scenario, TEMP = DBT/WBT, CAL = STANDARD/UKMO.
Each row is one year: the annual sum of the monthly values of a land cell (complete years only), averaged over the
cells of a zone with population weights. The years of the historical run come first, followed by those of each
scenario; all years in the files are used unless a range is given.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import INDICATORS

KEYS = ["SREGION", "GCM", "SSP", "RCP", "TEMP", "CAL", "YEAR"]
COLUMNS = KEYS + ["CDD"]
SORT = ["SREGION", "GCM", "RCP", "TEMP", "CAL", "YEAR"]


def gcm_name(model):
    """'gfdl-esm4' -> 'gfdlesm4' (AIM/Hub model label)."""
    return model.replace("-", "").replace("_", "")


class RegionalCDDTable:
    """Population-weighted degree days of every model, scenario and indicator for one comfort state.

    All indicators must be of one kind (CDD or HDD); it names the value column and the file.
    """

    def __init__(self, store, aggregator, models, scenarios, historical="historical", indicators=INDICATORS,
                 log=print, years=None):
        """years: optional (first, last) range of the table; None = every complete year in the files."""
        self.store = store
        self.aggregator = aggregator
        self.models = list(models)
        self.scenarios = list(scenarios)
        self.historical = historical
        self.indicators = tuple(indicators)
        self.years = None if years is None else (int(years[0]), int(years[1]))
        kinds = {i.kind for i in self.indicators}
        if len(kinds) != 1:
            raise ValueError(f"indicators of one kind expected, got {sorted(kinds)}")
        self.kind = kinds.pop()
        self.log = log or (lambda *a, **k: None)

    def build(self, state):
        t0 = time.time()
        parts = []
        for model in self.models:
            for ind in self.indicators:
                hist = self.store.read_annual(model, self.historical, ind, state) if self.historical else None
                for scen in self.scenarios:
                    series = self.store.read_annual(model, scen, ind, state)
                    if hist is not None:
                        series = hist.join(series)
                    if self.years is not None:
                        series = series.select(*self.years)
                    res = self.aggregator.means(series.values, series.lat, series.lon)
                    n = len(series.years)
                    parts.append(pd.DataFrame({
                        "SREGION": np.repeat(list(res), n), "GCM": gcm_name(model), "SSP": scen[:4].upper(),
                        "RCP": scen, "TEMP": ind.temperature, "CAL": ind.method.upper(),
                        "YEAR": np.tile(series.years, len(res)),
                        self.kind: np.concatenate(list(res.values()))}))
            self.log(f"  {state.label} {model} done ({time.time() - t0:.0f} s)")
        return pd.concat(parts, ignore_index=True).sort_values(SORT)[KEYS + [self.kind]].reset_index(drop=True)

    def write(self, state, out_dir, name_template="{kind}_{state}.csv"):
        """Write the table of `state` to out_dir/<name>; name_template placeholders: {kind}, {state} (e.g. 22C_60p),
        {temp}, {rh}."""
        df = self.build(state)
        target = Path(out_dir) / state.csv_name_of(self.kind, name_template)
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False, float_format="%.10g", lineterminator="\n")
        self.log(f"{target.name}: {len(df)} rows, {df.SREGION.nunique()} regions")
        return target
