"""Regular latitude-longitude grid (north-up), by default the global 0.5 deg ISIMIP/W5E5 grid."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Cell-centre grid: row 0 is the northernmost row (centre `north`), column 0 the westernmost (centre `west`)."""

    north: float = 89.75
    west: float = -179.75
    res: float = 0.5
    nlat: int = 360
    nlon: int = 720

    @property
    def shape(self):
        return (self.nlat, self.nlon)

    @property
    def lat(self):
        return self.north - self.res * np.arange(self.nlat)

    @property
    def lon(self):
        return self.west + self.res * np.arange(self.nlon)

    def rc(self, lat, lon):
        """Row and column of cell centres (arrays of lat/lon)."""
        r = np.round((self.north - np.asarray(lat, dtype=np.float64)) / self.res).astype(int)
        c = np.round((np.asarray(lon, dtype=np.float64) - self.west) / self.res).astype(int)
        if (r < 0).any() or (r >= self.nlat).any() or (c < 0).any() or (c >= self.nlon).any():
            raise ValueError("points outside the grid")
        return r, c

    def check(self, array, what="array"):
        if np.shape(array) != self.shape:
            raise ValueError(f"{what}: shape {np.shape(array)}, expected {self.shape}")
        return array

    @classmethod
    def from_coords(cls, lat, lon):
        """Grid of 1-D cell-centre coordinates (latitude may be north-first or south-first)."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        res = float(abs(lon[1] - lon[0])) if len(lon) > 1 else float(abs(lat[1] - lat[0]))
        return cls(north=float(lat.max()), west=float(lon.min()), res=res, nlat=len(lat), nlon=len(lon))


GLOBAL_05 = Grid()


def block_sum(a, k):
    """Sum non-overlapping k x k blocks of a 2-D array (30 arc-second pixels -> 0.5 deg: k = 60).
    Trailing rows/columns that do not fill a block are dropped."""
    a = np.asarray(a, dtype=np.float64)
    r, c = a.shape[0] // k, a.shape[1] // k
    return a[: r * k, : c * k].reshape(r, k, c, k).sum(axis=(1, 3))


def read_esri_ascii(path):
    """ESRI ASCII grid -> (meta, array). Used for the AIM/Hub country and population grids."""
    with open(path, "r") as f:
        meta = {}
        for _ in range(6):
            key, value = f.readline().split()[:2]
            meta[key.lower()] = float(value)
        data = np.loadtxt(f)
    meta["ncols"], meta["nrows"] = int(meta["ncols"]), int(meta["nrows"])
    return meta, data


def grid_of_esri_meta(meta):
    cell = meta["cellsize"]
    x0 = meta.get("xllcorner", meta.get("xllcenter", 0.0) - cell / 2)
    y0 = meta.get("yllcorner", meta.get("yllcenter", 0.0) - cell / 2)
    return Grid(north=y0 + cell * (meta["nrows"] - 0.5), west=x0 + cell / 2, res=cell,
                nlat=meta["nrows"], nlon=meta["ncols"])
