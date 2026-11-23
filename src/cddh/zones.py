"""Country of each grid cell and groupings of countries (ISO3 countries, AIM/Hub regions, study regions).

CountryGrid : UN M49 country code per cell + a country table (un_code, iso3[, country, <region columns>]).
Zones       : label per cell (e.g. ISO3 code or AIM/Hub region) and optional composite groups
              (e.g. 'Global North' = several regions, 'World' = all labels).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import GLOBAL_05, Grid, grid_of_esri_meta, read_esri_ascii


class CountryGrid:
    """Country code per grid cell and a table describing the countries.

    codes : int array on `grid`, UN M49 code per cell (0 or negative = no country)
    table : DataFrame with at least columns un_code and iso3; further columns map countries to groups
            (e.g. aim_region)
    """

    def __init__(self, codes, table, grid=GLOBAL_05):
        self.grid = grid
        self.codes = np.asarray(grid.check(codes, "country codes"), dtype=np.int64)
        self.table = pd.DataFrame(table).copy()
        for col in ("un_code", "iso3"):
            if col not in self.table:
                raise KeyError(f"country table needs column '{col}'")
        self.table["un_code"] = self.table["un_code"].astype(int)
        if self.table.un_code.duplicated().any():
            raise ValueError("duplicate un_code in the country table")

    @classmethod
    def load(cls, grid_path, table_path, grid=GLOBAL_05):
        """Country grid from .npz (key 'n_code'), .npy or an ESRI ASCII grid, and a CSV table."""
        grid_path = Path(grid_path)
        if grid_path.suffix == ".npz":
            z = np.load(grid_path)
            codes = z["n_code"]
            if "lat" in z and "lon" in z:
                grid = Grid.from_coords(z["lat"], z["lon"])
        elif grid_path.suffix == ".npy":
            codes = np.load(grid_path)
        else:
            meta, codes = read_esri_ascii(grid_path)
            grid = grid_of_esri_meta(meta)
            codes = np.where(codes == meta.get("nodata_value", -9999.0), 0, codes)
        return cls(np.asarray(codes).astype(np.int64), pd.read_csv(table_path), grid)

    def save(self, grid_path, table_path=None):
        Path(grid_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(grid_path, n_code=self.codes.astype(np.int16), lat=self.grid.lat, lon=self.grid.lon)
        if table_path:
            self.table.to_csv(table_path, index=False)

    def mapping(self, column):
        """{un_code: value of `column`} for countries with a non-empty value."""
        t = self.table.dropna(subset=[column])
        return {int(k): str(v) for k, v in zip(t.un_code, t[column]) if str(v) != ""}

    def labels(self, column="iso3"):
        """Label grid: value of `column` for the country of each cell ('' where none)."""
        lut = self.mapping(column)
        flat = self.codes.ravel()
        uniq, inv = np.unique(flat, return_inverse=True)
        lab = np.array([lut.get(int(u), "") for u in uniq], dtype=object)
        return lab[inv].reshape(self.codes.shape).astype(str)

    def zones(self, column="iso3", composites=None):
        """Zones of the countries grouped by `column` ('iso3' = one zone per country)."""
        return Zones(self.labels(column), self.grid, composites)


class Zones:
    """Label per grid cell plus composite groups {name: [labels] or '*' (all labels)}."""

    def __init__(self, labels, grid=GLOBAL_05, composites=None):
        self.grid = grid
        self.labels = np.asarray(grid.check(labels, "zone labels")).astype(str)
        self.composites = dict(composites or {})

    @classmethod
    def from_codes(cls, codes, names, grid=GLOBAL_05, composites=None):
        """Integer zone grid (0 = none, k = names[k-1]), e.g. a nine-region study grid."""
        codes = np.asarray(codes).astype(int)
        lut = np.array([""] + list(names), dtype=object)
        return cls(lut[np.clip(codes, 0, len(names))].astype(str), grid, composites)

    def groups(self):
        """{group name: set of labels}: every label on its own, then the composites."""
        present = sorted(set(np.unique(self.labels)) - {""})
        out = {k: {k} for k in present}
        for name, members in self.composites.items():
            out[name] = set(present) if members == "*" else set(members)
        return out

    def at(self, rows, cols):
        return self.labels[rows, cols]


def read_aim_region_map(path):
    """AIM/Hub region.map (tab-separated, ISO3 in column 2 and region in column 4) -> {iso3: region}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            if len(parts) >= 4 and parts[1] and parts[3]:
                out[parts[1]] = parts[3]
    return out


def country_table_from_aim(country_code_csv, region_map=None, extra=None):
    """Country table from AIM/Hub country_code.txt (Unsdcode, Iso3v10, ...) and region.map.

    extra: optional rows to add, e.g. [{'un_code': 158, 'iso3': 'TWN', 'aim_region': 'TWN'}]."""
    c = pd.read_csv(country_code_csv)
    t = pd.DataFrame({"un_code": c["Unsdcode"].astype(int), "iso3": c["Iso3v10"].astype(str)})
    name_col = next((k for k in c.columns if k.lower().startswith("country")), None)
    if name_col:
        t["country"] = c[name_col]
    if region_map is not None:
        regions = read_aim_region_map(region_map) if not isinstance(region_map, dict) else region_map
        t["aim_region"] = t.iso3.map(regions)
    if extra:
        t = pd.concat([t[~t.un_code.isin([int(r["un_code"]) for r in extra])], pd.DataFrame(extra)], ignore_index=True)
    return t.drop_duplicates("un_code").sort_values("iso3").reset_index(drop=True)


__all__ = ["CountryGrid", "Zones", "Grid", "read_aim_region_map", "country_table_from_aim"]
