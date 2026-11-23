"""Synthetic inputs on a small 4 x 5 grid (0.5 deg, 8.75-10.25 N, 0.25-2.25 E): no real data are needed."""
import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from cddh import Grid, LandMask, PopulationGrid
from cddh.zones import CountryGrid

GRID = Grid(north=10.25, west=0.25, res=0.5, nlat=4, nlon=5)
MODEL = "toy-esm1"
VARS = ("tas", "tasmax", "tasmin", "hurs")


def synthetic_fields(dates, grid=GRID, seed=0):
    """Daily tas/tasmax/tasmin (K) and hurs (%) with a seasonal cycle, float32, shape (days, lat, lon)."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    doy = np.asarray(dates.dayofyear, dtype=float)[:, None, None]
    lat = grid.lat[None, :, None]
    base = 273.15 + 24 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) - 0.8 * (lat - 9.5)
    base = base + rng.normal(0, 2, (n, grid.nlat, grid.nlon))
    rng_amp = 4 + rng.uniform(0, 4, (n, grid.nlat, grid.nlon))
    hurs = np.clip(60 + 25 * np.cos(2 * np.pi * doy / 365) + rng.normal(0, 8, (n, grid.nlat, grid.nlon)), 5, 99)
    return {"tas": base.astype(np.float32), "tasmax": (base + rng_amp).astype(np.float32),
            "tasmin": (base - rng_amp).astype(np.float32), "hurs": hurs.astype(np.float32)}


def write_daily(path, var, dates, data, grid=GRID):
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", grid.nlat)
        ds.createDimension("lon", grid.nlon)
        t = ds.createVariable("time", "f8", ("time",))
        t.units, t.calendar = "days since 1850-01-01 00:00:00", "proleptic_gregorian"
        t[:] = nc.date2num(dates.to_pydatetime(), t.units, t.calendar)
        ds.createVariable("lat", "f8", ("lat",))[:] = grid.lat
        ds.createVariable("lon", "f8", ("lon",))[:] = grid.lon
        v = ds.createVariable(var, "f4", ("time", "lat", "lon"), fill_value=np.float32(1e20))
        v[:] = data


def make_archive(root, scenario, periods, seed=0):
    """ISIMIP3b-like files root/<model>/<scenario>/<var>/..._<y0>_<y1>.nc; returns {var: full daily array}."""
    full = {v: [] for v in VARS}
    for i, (y0, y1) in enumerate(periods):
        dates = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
        f = synthetic_fields(dates, seed=seed + i)
        for v in VARS:
            write_daily(root / MODEL / scenario / v / f"{MODEL}_r1i1p1f1_w5e5_{scenario}_{v}_global_daily_{y0}_{y1}.nc",
                        v, dates, f[v])
            full[v].append(f[v])
    return {v: np.concatenate(a) for v, a in full.items()}


@pytest.fixture
def land():
    """Land = all cells except the north-east corner (ocean) and one coastal cell."""
    mask = np.ones(GRID.shape, dtype=bool)
    mask[0, 4] = False
    mask[3, 0] = False
    iso = np.where(np.arange(GRID.nlon)[None, :].repeat(GRID.nlat, 0)[mask] < 3, "AAA", "BBB")
    return LandMask(GRID.lat, GRID.lon, mask, iso)


@pytest.fixture
def countries():
    """Country AAA (code 1) in columns 0-2, BBB (code 2) in columns 3-4, CCC (code 3) in cell (3, 0) (off the land
    grid), no country in the ocean cell (0, 4). AAA and CCC form region R1, BBB region R2."""
    codes = np.where(np.arange(GRID.nlon)[None, :] < 3, 1, 2).repeat(GRID.nlat, 0)
    codes[0, 4] = 0
    codes[3, 0] = 3
    table = pd.DataFrame({"un_code": [1, 2, 3], "iso3": ["AAA", "BBB", "CCC"], "aim_region": ["R1", "R2", "R1"]})
    return CountryGrid(codes, table, GRID)


@pytest.fixture
def population():
    pop = np.arange(1, GRID.nlat * GRID.nlon + 1, dtype=float).reshape(GRID.shape) * 100
    pop[1, 1] = 0.0
    return PopulationGrid(pop, GRID)
