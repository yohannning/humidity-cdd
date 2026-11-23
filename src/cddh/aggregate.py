"""Population-weighted means of land-cell CDD over countries, regions and composite groups.

    CDD_group = sum_i pop_i * CDD_i / sum_i pop_i     over the cells i of the group with pop_i > 0

Two ways of choosing the cells i:
  fill_offgrid=True  (default; regional CDD tables): every populated cell with a zone label. Cells that are not on
                     the land grid of the CDD files (coasts, small islands) take the value of the nearest land cell.
  fill_offgrid=False (decomposition): only the land cells of the CDD files, each with its own population.
"""
import warnings

import numpy as np
import pandas as pd


def weighted_mean_by_group(values, groups, weights):
    """Weighted mean of `values` (..., n) within each group label; cells with weight <= 0 or non-finite values
    are ignored. Returns {group: array of shape values.shape[:-1]}."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values).all(axis=tuple(range(values.ndim - 1))) if values.ndim > 1 else np.isfinite(values)
    keep = (weights > 0) & finite
    out = {}
    for g in np.unique(groups[keep]):
        m = keep & (groups == g)
        w = weights[m]
        out[g] = (values[..., m] * w).sum(axis=-1) / w.sum()
    return out


class NearestLand:
    """For populated grid cells, the index of the land cell whose CDD they take (itself when it is a land cell).

    rows, cols   : grid cells (row, column) covered by the mapping
    land_index   : index into the land cells (order of the CDD files) for each of these cells
    land_row/col : row and column of every land cell, to check that a mapping fits the land cells of the CDD files
    """

    def __init__(self, rows, cols, land_index, land_row, land_col):
        self.rows, self.cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        self.land_index = np.asarray(land_index, dtype=np.int64)
        self.land_row, self.land_col = np.asarray(land_row, dtype=np.int64), np.asarray(land_col, dtype=np.int64)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["row"], z["col"], z["land_index"], z["land_row"], z["land_col"])

    def save(self, path):
        np.savez_compressed(path, row=self.rows.astype(np.int16), col=self.cols.astype(np.int16),
                            land_index=self.land_index.astype(np.int32), land_row=self.land_row.astype(np.int16),
                            land_col=self.land_col.astype(np.int16))
        return path

    @classmethod
    def compute(cls, land_row, land_col, rows, cols):
        """Nearest land cell in grid units (row/column distance). Equidistant land cells are resolved by taking the one
        with the smallest index, so the result does not depend on the k-d tree implementation."""
        from scipy.spatial import cKDTree

        land_row, land_col = np.asarray(land_row), np.asarray(land_col)
        pts = np.c_[rows, cols].astype(np.float64)
        k = min(16, len(land_row))
        d, j = cKDTree(np.c_[land_row, land_col].astype(np.float64)).query(pts, k=k)
        d, j = d.reshape(len(pts), k), j.reshape(len(pts), k)
        tie = np.isclose(d, d[:, :1], rtol=0, atol=1e-9)
        best = np.where(tie, j, np.iinfo(np.int64).max).min(axis=1)
        return cls(rows, cols, best, land_row, land_col)

    def fits(self, land_row, land_col):
        return np.array_equal(self.land_row, land_row) and np.array_equal(self.land_col, land_col)

    def lookup(self, rows, cols, nlon):
        key = self.rows * nlon + self.cols
        order = np.argsort(key)
        want = np.asarray(rows) * nlon + np.asarray(cols)
        pos = np.searchsorted(key[order], want)
        pos = np.minimum(pos, len(key) - 1)
        ok = key[order][pos] == want
        if not ok.all():
            raise KeyError(f"{int((~ok).sum())} populated cells are not covered by the nearest-land mapping")
        return self.land_index[order][pos]


class Aggregator:
    """Population-weighted aggregation of land-cell values to the groups of `zones`.

    nearest: optional NearestLand mapping used for the populated cells that are not land cells (fill_offgrid=True),
    e.g. data/land_cells/nearest_land_05deg.npz, the mapping of the CDD tables of the paper. Without it the nearest
    land cell is computed (NearestLand.compute)."""

    def __init__(self, population, zones, fill_offgrid=True, min_value=None, nearest=None):
        if population.grid != zones.grid:
            raise ValueError("population and zones must be on the same grid")
        self.population = population
        self.zones = zones
        self.grid = zones.grid
        self.fill_offgrid = fill_offgrid
        self.min_value = min_value
        self.nearest = nearest
        self._cache = {}

    def cells(self, lat, lon):
        """(index into the land cells, weight, zone label) of the cells that enter the means."""
        key = (np.asarray(lat).tobytes(), np.asarray(lon).tobytes())
        if key not in self._cache:
            r, c = self.grid.rc(lat, lon)
            pop, lab = self.population.values, self.zones.labels
            if self.fill_offgrid:
                rr, cc = np.nonzero((lab != "") & (pop > 0))
                if self.nearest is not None and self.nearest.fits(r, c):
                    j = self.nearest.lookup(rr, cc, self.grid.nlon)
                else:
                    if self.nearest is not None:
                        warnings.warn("the nearest-land mapping was made for other land cells; computing it instead")
                    j = NearestLand.compute(r, c, rr, cc).land_index
            else:
                j = np.flatnonzero((lab[r, c] != "") & (pop[r, c] > 0))
                rr, cc = r[j], c[j]
            self._cache[key] = (np.asarray(j), pop[rr, cc], lab[rr, cc])
        return self._cache[key]

    def means(self, values, lat, lon):
        """{group: population-weighted mean} of values (..., land cells); groups = zones.groups()."""
        j, w, lab = self.cells(lat, lon)
        v = np.asarray(values, dtype=np.float64)[..., j]
        out = {}
        for name, members in self.zones.groups().items():
            labels = np.where(np.isin(lab, list(members)), name, "")
            res = weighted_mean_by_group(v, labels, w * (labels != ""))
            if name in res:
                out[name] = res[name]
        if self.min_value is not None:
            out = {k: np.where(x < self.min_value, 0.0, x) for k, x in out.items()}
        return out

    def table(self, annual, label="group"):
        """Long DataFrame (label, year, value) of an AnnualCDD."""
        res = self.means(annual.values, annual.lat, annual.lon)
        return pd.DataFrame({label: np.repeat(list(res), len(annual.years)),
                             "year": np.tile(annual.years, len(res)),
                             "value": np.concatenate(list(res.values())) if res else []})
