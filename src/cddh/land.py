"""Land cells on which CDD is computed.

The land cells of the original runs (58,841 cells of the 0.5 deg grid, without Antarctica, each with the AIM/Hub
region or country label of the polygon containing its centre) are bundled as data/land_cells/land_cells_05deg.npz.
LandMask.from_shapefile builds a land mask from any country layer instead.
"""
import numpy as np


class LandMask:
    """Boolean (lat, lon) mask of land cells on a climate grid, with the ISO3 code of each land cell.

    Land cells are ordered as in the flattened (lat, lon) grid of the climate files; the same order is used by the
    monthly CDD files (dimension 'grid').
    """

    def __init__(self, lat, lon, mask, iso):
        self.lat_axis = np.asarray(lat, dtype=np.float64)
        self.lon_axis = np.asarray(lon, dtype=np.float64)
        self.mask = np.asarray(mask, dtype=bool).reshape(len(self.lat_axis), len(self.lon_axis))
        self.iso = np.asarray(iso, dtype=str)
        if self.iso.shape != (int(self.mask.sum()),):
            raise ValueError("one ISO code per land cell expected")

    @property
    def n(self):
        return int(self.mask.sum())

    @property
    def lat(self):
        """Latitude of each land cell."""
        return np.broadcast_to(self.lat_axis[:, None], self.mask.shape)[self.mask]

    @property
    def lon(self):
        """Longitude of each land cell."""
        return np.broadcast_to(self.lon_axis[None, :], self.mask.shape)[self.mask]

    @classmethod
    def from_shapefile(cls, path, lat, lon, iso_column="iso_a3", exclude=()):
        """Cells whose centre lies within a polygon of a country layer (e.g. Natural Earth 1:110m admin-0 countries).
        Polygons with code '-99', an empty code or a code in `exclude` (e.g. 'ATA') are ignored. Needs geopandas."""
        import geopandas as gpd

        world = gpd.read_file(path)
        world = world.set_crs(4326) if world.crs is None else world.to_crs(4326)
        world = world[world.geometry.notna() & ~world[iso_column].isin(list(exclude))]
        lat2d, lon2d = np.meshgrid(np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64), indexing="ij")
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()), crs="EPSG:4326")
        joined = points.sjoin(world[[iso_column, "geometry"]], how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        iso = joined[iso_column].fillna("").astype(str).to_numpy()
        iso = np.where((iso == "-99") | (iso == "nan"), "", iso)
        mask = iso != ""
        return cls(lat, lon, mask, iso[mask])

    def save(self, path):
        """Save the land cells (lat, lon, label) as .npz."""
        np.savez_compressed(path, lat=self.lat.astype(np.float32), lon=self.lon.astype(np.float32),
                            label=self.iso.astype("U3"))
        return path

    @classmethod
    def load(cls, path, lat, lon):
        """Land cells saved with `save` (or data/land_cells/land_cells_05deg.npz) on the climate grid lat/lon."""
        z = np.load(path)
        return cls.from_cells(lat, lon, z["lat"], z["lon"], z["label"])

    @classmethod
    def from_cells(cls, lat, lon, cell_lat, cell_lon, iso=None):
        """Mask from the coordinates of the land cells (e.g. as stored in a monthly CDD file)."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        i = np.abs(lat[:, None] - np.asarray(cell_lat)[None, :]).argmin(axis=0)
        j = np.abs(lon[:, None] - np.asarray(cell_lon)[None, :]).argmin(axis=0)
        mask = np.zeros((len(lat), len(lon)), dtype=bool)
        mask[i, j] = True
        order = np.argsort(i * len(lon) + j)
        if not np.array_equal(order, np.arange(len(order))):
            raise ValueError("land cells must be in (lat, lon) row-major order")
        iso = np.full(len(i), "", dtype="<U3") if iso is None else np.asarray(iso, dtype=str)
        return cls(lat, lon, mask, iso)
