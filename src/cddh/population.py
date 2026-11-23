"""Population weights on the 0.5 deg grid."""
from pathlib import Path

import numpy as np

from .grid import GLOBAL_05, block_sum, grid_of_esri_meta, read_esri_ascii


class PopulationGrid:
    """Persons per grid cell (north-up array on `grid`)."""

    def __init__(self, values, grid=GLOBAL_05, name="population"):
        self.grid = grid
        self.values = np.asarray(grid.check(values, name), dtype=np.float64)
        self.name = name

    @property
    def total(self):
        return float(self.values.sum())

    def at(self, lat, lon):
        r, c = self.grid.rc(lat, lon)
        return self.values[r, c]

    def save(self, path):
        """Save as .npy (array only) or .npz (array + grid)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            g = self.grid
            np.savez_compressed(path, pop=self.values, north=g.north, west=g.west, res=g.res)
        else:
            np.save(path, self.values)
        return path

    @classmethod
    def load(cls, path, grid=GLOBAL_05, name=None):
        """Load from .npy (array on `grid`), .npz (key 'pop') or an ESRI ASCII grid (.txt/.asc)."""
        path = Path(path)
        name = name or path.stem
        if path.suffix == ".npy":
            return cls(np.load(path), grid, name)
        if path.suffix == ".npz":
            z = np.load(path)
            g = grid
            if "north" in z:
                g = type(grid)(float(z["north"]), float(z["west"]), float(z["res"]), *z["pop"].shape)
            return cls(z["pop"], g, name)
        meta, data = read_esri_ascii(path)
        data = np.where(data == meta.get("nodata_value", -9999.0), 0.0, data)
        return cls(np.maximum(data, 0.0), grid_of_esri_meta(meta), name)

    @classmethod
    def from_raster(cls, path, grid=GLOBAL_05, name=None):
        """Sum a fine global population raster (e.g. WorldPop 2020 1 km, 30 arc-seconds) into the cells of `grid`.

        The raster is read one row of grid cells at a time; negative, non-finite and nodata pixels count as 0.
        Needs rasterio."""
        import rasterio
        from rasterio.windows import from_bounds

        out = np.zeros(grid.shape)
        north_edge = grid.north + grid.res / 2
        west_edge = grid.west - grid.res / 2
        east_edge = west_edge + grid.res * grid.nlon
        with rasterio.open(path) as src:
            for i in range(grid.nlat):
                top = north_edge - i * grid.res
                win = from_bounds(west_edge, top - grid.res, east_edge, top, src.transform)
                win = win.round_offsets().round_lengths()
                a = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
                a[(a < 0) | ~np.isfinite(a)] = 0
                if src.nodata is not None:
                    a[a == src.nodata] = 0
                k = a.shape[1] // grid.nlon
                out[i] = a[:, : k * grid.nlon].reshape(a.shape[0], grid.nlon, k).sum(axis=(0, 2))
        return cls(out, grid, name or Path(path).stem)

    @classmethod
    def from_fine_array(cls, array, factor, grid=GLOBAL_05, name="population"):
        """Sum a fine array already aligned with `grid` (factor x factor pixels per cell)."""
        return cls(block_sum(np.where(np.isfinite(array) & (array > 0), array, 0.0), factor), grid, name)
