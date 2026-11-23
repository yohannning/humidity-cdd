"""Monthly degree days (CDD and HDD) on land cells from daily climate files, for several indicators and comfort
states in one pass."""
import time

import numpy as np
import pandas as pd

from .climate import DailyFields
from .indicators import INDICATORS, STATES, temperature


class CDDCalculator:
    """Compute monthly degree-day sums for every (indicator, comfort state) combination, reading each input file
    once. Works for CDD and HDD indicators alike (e.g. indicators=HDD_INDICATORS, states=HDD_BASES).

    Arithmetic is float32 as in the climate files; each month is read in one piece and its daily CDD summed
    into the monthly total of every cell.
    """

    def __init__(self, land, indicators=INDICATORS, states=STATES, log=print, jobs=None):
        """jobs: optional explicit list of (indicator, state) pairs instead of all indicators x states."""
        self.land = land
        self.indicators = tuple(indicators)
        self.states = tuple(states)
        self.log = log or (lambda *a, **k: None)
        self._jobs = None if jobs is None else list(jobs)
        if self._jobs is not None:
            self.indicators = tuple(dict.fromkeys(i for i, _ in self._jobs))
            self.states = tuple(dict.fromkeys(s for _, s in self._jobs))

    @property
    def jobs(self):
        if self._jobs is not None:
            return list(self._jobs)
        return [(ind, st) for ind in self.indicators for st in self.states]

    @property
    def variables(self):
        return tuple(sorted({v for ind in self.indicators for v in ind.variables}))

    def accumulate(self, fields):
        """Monthly sums {(indicator, state): (n_cells,) float32} from daily fields {var: (days, n_cells)}."""
        cache, out = {}, {}
        for ind, st in self.jobs:
            for v in ind.temperature_variables:
                if (ind.temperature, v) not in cache:
                    cache[(ind.temperature, v)] = temperature(ind.temperature, fields, v)
            temps = {v: cache[(ind.temperature, v)] for v in ind.temperature_variables}
            out[(ind, st)] = ind.daily(temps, ind.threshold(st)).sum(axis=0).astype(np.float32)
        return out

    def monthly_from_fields(self, months):
        """Monthly CDD from an iterable of (month, fields) -> (DatetimeIndex, {(indicator, state): (months, cells)})."""
        index, rows = [], {job: [] for job in self.jobs}
        for month, fields in months:
            index.append(month)
            for job, v in self.accumulate(fields).items():
                rows[job].append(v)
        return pd.DatetimeIndex(index), {job: np.vstack(v) if v else np.zeros((0, self.land.n), np.float32)
                                         for job, v in rows.items()}

    def run(self, archive, model, scenario, years=None):
        """Monthly CDD for all periods of the archive that cover `years` (inclusive (y0, y1); None = all)."""
        t0 = time.time()
        index, parts = [], {job: [] for job in self.jobs}
        for y0, y1 in archive.periods(model, scenario, years, self.variables):
            paths = {v: archive.path(model, scenario, v, y0, y1) for v in self.variables}
            with DailyFields(paths, self.land.mask, years) as f:
                months, res = self.monthly_from_fields(f.months())
            index.append(months)
            for job, v in res.items():
                parts[job].append(v)
            self.log(f"  {model} {scenario} {y0}-{y1}: {len(months)} months, {len(self.jobs)} indicator x state "
                     f"({time.time() - t0:.0f} s)")
        months = index[0].append(index[1:]) if len(index) > 1 else index[0]
        if months.has_duplicates:
            raise ValueError("overlapping periods in the climate archive")
        return months, {job: np.vstack(v) for job, v in parts.items()}

    def run_to_store(self, archive, store, model, scenario, years=None, overwrite=False):
        """Compute and write one monthly file per (indicator, state); existing files are kept unless overwrite."""
        todo = [(i, s) for i, s in self.jobs if overwrite or not store.exists(model, scenario, i, s)]
        if not todo:
            self.log(f"{model} {scenario}: all files exist")
            return []
        calc = CDDCalculator(self.land, log=self.log, jobs=todo)
        months, res = calc.run(archive, model, scenario, years)
        written = []
        for ind, st in todo:
            written.append(store.write(model, scenario, ind, st, months, res[(ind, st)], self.land))
        self.log(f"{model} {scenario}: wrote {len(written)} files")
        return written

