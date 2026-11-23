"""Settings: where the inputs are and what to run.

Read from a TOML file (see config/cddh.example.toml; the default file is ./cddh.toml, or the file named by the
CDDH_CONFIG environment variable) and overridden by environment variables CDDH_<KEY> (e.g. CDDH_CLIMATE_DIR).
Relative paths in the file are relative to the file's folder. No location is built into the code.
"""
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

PATH_KEYS = ("climate_dir", "cdd_dir", "hdd_dir", "output_dir", "population", "country_grid", "country_table",
             "land_cells", "nearest_land", "land_shapefile", "study_regions", "study_region_names",
             "region_table_dir", "country_table_dir", "decomposition_dir")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Settings:
    climate_dir: Path = None          # daily climate archive (ISIMIP3b layout)
    climate_pattern: str = None       # filename pattern of the archive (default: ISIMIP3b)
    cdd_dir: Path = None              # monthly land-cell CDD files (written by `cddh cdd`)
    cdd_template: str = None          # file name template of the monthly files (default: store.DEFAULT_TEMPLATE)
    hdd_dir: Path = None              # monthly land-cell HDD files (default: cdd_dir)
    hdd_template: str = None          # file name template of the HDD files (default: cdd_template)
    output_dir: Path = Path("results")
    region_table_dir: Path = None     # regional tables (default: <output_dir>/{kind}_region); may contain {kind}
    country_table_dir: Path = None    # country tables (default: <output_dir>/{kind}_country); may contain {kind}
    table_name: str = "{kind}_{state}.csv"   # table file name; {kind}, {state} (22C_60p, 18C), {temp}, {rh}
    decomposition_dir: Path = None    # decomposition output (default: <output_dir>/decomposition)
    population: Path = None           # population grid (.npy/.npz/ESRI ASCII), e.g. from `cddh population`
    country_grid: Path = DATA_DIR / "country_grid" / "n_code_05deg.npz"
    country_table: Path = DATA_DIR / "country_grid" / "countries.csv"
    land_cells: Path = DATA_DIR / "land_cells" / "land_cells_05deg.npz"   # land cells of the original runs
    nearest_land: Path = DATA_DIR / "land_cells" / "nearest_land_05deg.npz"  # populated cell -> land cell (tables)
    land_shapefile: Path = None       # alternative: country polygons defining the land cells (used when set)
    study_regions: Path = DATA_DIR / "study_regions" / "region9_05deg.npy"
    study_region_names: Path = DATA_DIR / "study_regions" / "region9_names.txt"
    global_north: list = field(default_factory=lambda: [1, 2, 3, 4])   # study-region codes of the Global North
    models: list = field(default_factory=lambda: ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "ukesm1-0-ll"])
    scenarios: list = field(default_factory=lambda: ["ssp126", "ssp245", "ssp370", "ssp585"])
    historical: str = "historical"
    years: list = None                # [first, last] year to compute and tabulate (default: all years in the files)
    decomposition_periods: dict = None   # {name: [first, last]} with a 'base' period, e.g. base = [2010, 2019]
    hdd_bases: list = field(default_factory=lambda: [18.0])            # HDD base temperatures (C, dry-bulb)
    hdd_methods: list = field(default_factory=lambda: ["Standard", "UKMO"])   # HDD methods to compute

    @classmethod
    def load(cls, path=None, env=None):
        env = os.environ if env is None else env
        path = path or env.get("CDDH_CONFIG") or ("cddh.toml" if Path("cddh.toml").exists() else None)
        values, base = {}, Path.cwd()
        if path:
            path = Path(path)
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            base = path.resolve().parent
            for section in raw.values() if all(isinstance(v, dict) for v in raw.values()) else [raw]:
                values.update(section)
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise KeyError(f"unknown settings: {sorted(unknown)}")
        for k in known:
            if f"CDDH_{k.upper()}" in env:
                v = env[f"CDDH_{k.upper()}"]
                if k in ("models", "scenarios", "hdd_methods"):
                    v = v.split(",")
                elif k == "hdd_bases":
                    v = [float(x) for x in v.split(",")]
                elif k == "years":
                    v = [int(x) for x in v.replace("-", ",").split(",")]
                elif k == "decomposition_periods":
                    v = v.split(",")              # base=2010-2019,end=2090-2099
                values[k] = v
        for k in PATH_KEYS:
            if values.get(k) not in (None, ""):
                p = Path(values[k]).expanduser()
                values[k] = p if p.is_absolute() else base / p
        return cls(**values)

    def table_dir(self, level, kind):
        """Folder of the regional or country tables of `kind` (CDD/HDD)."""
        d = getattr(self, f"{level}_table_dir")
        d = Path(self.output_dir) / f"{{kind}}_{level}" if d is None else d
        return Path(str(d).format(kind=kind, level=level))

    def require(self, key):
        v = getattr(self, key)
        if v is None or not Path(v).exists():
            raise FileNotFoundError(f"setting '{key}' is not set or does not exist (set it in cddh.toml or "
                                    f"CDDH_{key.upper()})")
        return Path(v)
