"""Daily climate NetCDF files (ISIMIP3b bias-adjusted or GSWP3-W5E5 obsclim): locating files and reading land cells.

Files are found with a filename pattern relative to the archive root, with the placeholders
{model}, {scenario}, {var}, {y0}, {y1}. The default is the ISIMIP3b layout
    <root>/<model>/<scenario>/<var>/<model>_<member>_w5e5_<scenario>_<var>_global_daily_<y0>_<y1>.nc
and W5E5_PATTERN is the GSWP3-W5E5 obsclim layout (one folder with all files, model = scenario = 'obsclim').
"""
import re
from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

ISIMIP3B_PATTERN = "{model}/{scenario}/{var}/{model}_*_w5e5_{scenario}_{var}_global_daily_{y0}_{y1}.nc"
W5E5_PATTERN = "gswp3-w5e5_obsclim_{var}_global_daily_{y0}_{y1}.nc"
VARIABLES = ("tas", "tasmax", "tasmin", "hurs")


@dataclass(frozen=True)
class ClimateFile:
    var: str
    y0: int
    y1: int
    path: Path


class ClimateArchive:
    """Daily climate files of one archive (root folder + filename pattern)."""

    def __init__(self, root, pattern=ISIMIP3B_PATTERN):
        self.root = Path(root)
        self.pattern = pattern

    def files(self, model, scenario, var):
        """All files of a variable, sorted by start year."""
        glob = self.pattern.format(model=model, scenario=scenario, var=var, y0="[0-9][0-9][0-9][0-9]",
                                   y1="[0-9][0-9][0-9][0-9]")
        out = []
        for p in sorted(self.root.glob(glob)):
            y0, y1 = map(int, re.findall(r"_(\d{4})_(\d{4})\.nc$", p.name)[0])
            out.append(ClimateFile(var, y0, y1, p))
        if not out:
            raise FileNotFoundError(f"no {var} files for {model} {scenario} under the climate archive ({glob})")
        return sorted(out, key=lambda f: f.y0)

    def periods(self, model, scenario, years=None, variables=VARIABLES):
        """Periods (y0, y1) of the archive covering `years` (inclusive range), with every variable present."""
        spans = None
        for v in variables:
            s = {(f.y0, f.y1) for f in self.files(model, scenario, v)}
            spans = s if spans is None else spans & s
        spans = sorted(spans)
        if years is not None:
            spans = [(a, b) for a, b in spans if b >= years[0] and a <= years[1]]
        if not spans:
            raise FileNotFoundError(f"no common periods of {variables} for {model} {scenario}")
        return spans

    def path(self, model, scenario, var, y0, y1):
        for f in self.files(model, scenario, var):
            if (f.y0, f.y1) == (y0, y1):
                return f.path
        raise FileNotFoundError(f"{model} {scenario} {var} {y0}-{y1}")

    def coords(self, model, scenario, var="tas"):
        """Latitude and longitude of the archive grid (as stored in the files)."""
        with nc.Dataset(self.files(model, scenario, var)[0].path) as ds:
            return np.asarray(ds.variables["lat"][:], dtype=np.float64), np.asarray(ds.variables["lon"][:], dtype=np.float64)


def read_dates(ds):
    t = ds.variables["time"]
    d = nc.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
    return pd.DatetimeIndex([pd.Timestamp(x.year, x.month, x.day) for x in d])


class DailyFields:
    """Open daily files of several variables for one period and read whole months on selected cells.

    with DailyFields({'tas': path, 'hurs': path}, cells) as f:
        for month, fields in f.months():          # fields: {var: (days, n_cells) float32}
    `cells` is a boolean (lat, lon) mask or None (all cells).
    """

    def __init__(self, paths, cells=None, years=None):
        self.paths = dict(paths)
        self.cells = None if cells is None else np.asarray(cells, dtype=bool).ravel()
        self.years = years
        self._ds = {}

    def __enter__(self):
        self._ds = {v: nc.Dataset(p) for v, p in self.paths.items()}
        dates = None
        for v, ds in self._ds.items():
            d = read_dates(ds)
            if dates is None:
                dates = d
            elif len(d) != len(dates) or not (d == dates).all():
                raise ValueError(f"time axis of {v} differs from the other variables")
        self.dates = dates
        return self

    def __exit__(self, *exc):
        for ds in self._ds.values():
            ds.close()
        self._ds = {}

    def _read(self, var, a, b):
        data = np.asarray(np.ma.filled(self._ds[var].variables[var][a:b], np.nan), dtype=np.float32)
        data = data.reshape(b - a, -1)
        return data if self.cells is None else data[:, self.cells]

    def months(self):
        """Yield (month start Timestamp, {var: (days, n_cells) float32}) in time order."""
        keys = self.dates.year * 100 + self.dates.month
        for key in pd.unique(keys):
            y, m = divmod(int(key), 100)
            if self.years is not None and not (self.years[0] <= y <= self.years[1]):
                continue
            idx = np.flatnonzero(keys == key)
            a, b = int(idx[0]), int(idx[-1]) + 1
            if b - a != len(idx):
                raise ValueError(f"days of {y}-{m:02d} are not contiguous")
            yield pd.Timestamp(y, m, 1), {v: self._read(v, a, b) for v in self._ds}
