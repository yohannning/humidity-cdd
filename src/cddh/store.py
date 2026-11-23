"""Monthly land-cell degree-day files (CDD and HDD): naming, writing and reading.

One NetCDF file per model, scenario, indicator and comfort state (or HDD base temperature):
  time (months), grid (land cells); variables lat(grid), lon(grid), iso_a3(grid), CDD or HDD(time, grid) [degree days].
"""
from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

# File name relative to the store root. Placeholders: {kind} (CDD/HDD), {temperature} (DBT/WBT),
# {method} (Standard/UKMO), {model}, {scenario}, {tag} (e.g. base22C_rh60, base18C). Settable with cdd_template.
DEFAULT_TEMPLATE = "{model}/{scenario}/{kind}_{temperature}_{method}_{model}_{scenario}_{tag}_monthly.nc"
COMPRESS_LEVEL = 4


@dataclass
class AnnualCDD:
    """Annual CDD on land cells: values (years, cells)."""

    years: np.ndarray
    values: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    def join(self, other):
        """Append a later period on the same land cells (e.g. historical + SSP)."""
        if not (np.array_equal(self.lat, other.lat) and np.array_equal(self.lon, other.lon)):
            raise ValueError("land cells differ")
        if other.years[0] <= self.years[-1]:
            raise ValueError("periods overlap")
        return AnnualCDD(np.concatenate([self.years, other.years]), np.vstack([self.values, other.values]),
                         self.lat, self.lon)

    def select(self, y0, y1):
        m = (self.years >= y0) & (self.years <= y1)
        return AnnualCDD(self.years[m], self.values[m], self.lat, self.lon)

    def mean(self, y0, y1):
        """Mean annual CDD per cell over the years y0..y1 (all of them must be present)."""
        sel = self.select(y0, y1)
        if len(sel.years) != y1 - y0 + 1:
            raise ValueError(f"years {y0}-{y1} not all available")
        return sel.values.mean(axis=0)


class CDDStore:
    """Folder of monthly land-cell degree-day files (CDD and HDD)."""

    def __init__(self, root, template=DEFAULT_TEMPLATE):
        self.root = Path(root)
        self.template = template

    def path(self, model, scenario, indicator, state):
        return self.root / self.template.format(
            model=model, scenario=scenario, kind=indicator.kind, temperature=indicator.temperature, method=indicator.method,
            tag=state.tag)

    def exists(self, model, scenario, indicator, state):
        return self.path(model, scenario, indicator, state).exists()

    def write(self, model, scenario, indicator, state, months, values, land, attrs=None):
        """Write monthly degree days (months, land cells) with the land cell coordinates and ISO codes of `land`."""
        path = self.path(model, scenario, indicator, state)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".nc.tmp")
        months = pd.DatetimeIndex(months)
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(months), land.n):
            raise ValueError(f"values {values.shape}, expected {(len(months), land.n)}")
        with nc.Dataset(tmp, "w", format="NETCDF4_CLASSIC") as ds:
            ds.createDimension("time", len(months))
            ds.createDimension("grid", land.n)
            ds.createDimension("iso_strlen", 3)
            t = ds.createVariable("time", "f8", ("time",))
            t.units = "days since 1900-01-01 00:00:00"
            t.calendar = "proleptic_gregorian"
            t[:] = nc.date2num(months.to_pydatetime(), units=t.units, calendar=t.calendar)
            ds.createVariable("grid", "i4", ("grid",))[:] = np.arange(land.n, dtype=np.int32)
            for name, v in (("lat", land.lat), ("lon", land.lon)):
                var = ds.createVariable(name, "f4", ("grid",), zlib=True, complevel=COMPRESS_LEVEL)
                var[:] = v.astype(np.float32)
                var.units = "degrees_north" if name == "lat" else "degrees_east"
            iso = ds.createVariable("iso_a3", "S1", ("grid", "iso_strlen"), zlib=True, complevel=COMPRESS_LEVEL)
            iso.set_auto_chartostring(False)
            iso[:, :] = np.asarray(land.iso, dtype="S3").view("S1").reshape(land.n, 3)
            cdd = ds.createVariable(indicator.kind, "f4", ("time", "grid"), zlib=True, complevel=COMPRESS_LEVEL, shuffle=True,
                                    chunksizes=(min(12, len(months)), min(20000, land.n)))
            cdd.units = "degree_days"
            cdd.long_name = f"Monthly {indicator.kind} on land grid cells, {indicator.name}"
            cdd[:, :] = values
            ds.model = model
            ds.scenario = scenario
            ds.indicator = indicator.name
            ds.temperature_index = indicator.temperature
            ds.degree_day_method = indicator.method
            ds.degree_day_type = indicator.kind
            ds.base_temperature_celsius = state.temp_c
            if state.rh_pct is not None:
                ds.base_relative_humidity_percent = state.rh_pct
            ds.setncattr(f"{indicator.kind.lower()}_threshold_celsius", indicator.threshold(state))
            for k, v in (attrs or {}).items():
                ds.setncattr(k, v)
        tmp.replace(path)
        return path

    def read_monthly(self, model, scenario, indicator, state):
        """(months DatetimeIndex, values (months, cells) float64, lat, lon) of one file."""
        path = self.path(model, scenario, indicator, state)
        with nc.Dataset(path) as ds:
            t = ds.variables["time"]
            d = nc.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
            months = pd.DatetimeIndex([pd.Timestamp(x.year, x.month, 1) for x in d])
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            cdd = np.asarray(ds.variables[indicator.kind][:], dtype=np.float64)
        return months, cdd, lat, lon

    def read_annual(self, model, scenario, indicator, state):
        """Annual sums of complete years."""
        months, cdd, lat, lon = self.read_monthly(model, scenario, indicator, state)
        years = np.unique(months.year)
        out = np.zeros((len(years), cdd.shape[1]))
        for i, y in enumerate(years):
            m = months.year == y
            if m.sum() != 12:
                raise ValueError(f"{self.path(model, scenario, indicator, state).name}: {m.sum()} months in {y}")
            out[i] = cdd[m].sum(axis=0)
        return AnnualCDD(years, out, lat, lon)

    def read_series(self, model, scenarios, indicator, state):
        """Annual CDD of consecutive scenarios joined in time, e.g. ('historical', 'ssp245')."""
        series = None
        for s in scenarios:
            a = self.read_annual(model, s, indicator, state)
            series = a if series is None else series.join(a)
        return series
