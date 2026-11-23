"""End-to-end run of the command line on synthetic files, settings handling, bundled data and repository hygiene."""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cddh import CountryGrid, LandMask, Settings
from cddh.cli import main

from conftest import GRID, MODEL, make_archive

ROOT = Path(__file__).resolve().parents[1]


def test_settings_file_and_environment(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text('[paths]\nclimate_dir = "clim"\n[run]\nmodels = ["m1"]\n', encoding="utf-8")
    s = Settings.load(f, env={"CDDH_SCENARIOS": "ssp126,ssp585"})
    assert s.climate_dir == tmp_path / "clim" and s.models == ["m1"] and s.scenarios == ["ssp126", "ssp585"]
    assert s.hdd_bases == [18.0] and s.hdd_methods == ["Standard", "UKMO"]
    assert s.table_dir("region", "HDD") == s.output_dir / "HDD_region"
    assert s.years is None and s.decomposition_periods is None
    s = Settings.load(f, env={"CDDH_YEARS": "2001-2050", "CDDH_DECOMPOSITION_PERIODS": "base=2001-2010,end=2041-2050"})
    assert s.years == [2001, 2050] and s.decomposition_periods == ["base=2001-2010", "end=2041-2050"]
    s = Settings.load(f, env={"CDDH_HDD_BASES": "18,15.5", "CDDH_HDD_METHODS": "UKMO"})
    assert s.hdd_bases == [18.0, 15.5] and s.hdd_methods == ["UKMO"]
    assert Settings.load(f, env={"CDDH_CLIMATE_DIR": str(tmp_path / "x")}).climate_dir == tmp_path / "x"
    f.write_text('typo_key = 1\n', encoding="utf-8")
    with pytest.raises(KeyError):
        Settings.load(f, env={})
    with pytest.raises(FileNotFoundError):
        Settings().require("climate_dir")


def test_output_locations_are_settings(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text("""
[paths]
cdd_dir = "dd"
hdd_dir = "heating"
region_table_dir = "tables/{kind}_by_region"
[files]
cdd_template = "{kind}/{model}_{scenario}_{temperature}_{method}_{tag}.nc"
table_name = "{kind}{temp}_{rh}.csv"
""", encoding="utf-8")
    from cddh import HDD_UKMO, WBT_UKMO, ComfortState
    from cddh.cli import store_of
    s = Settings.load(f, env={})
    assert s.table_dir("region", "CDD") == tmp_path / "tables" / "CDD_by_region"
    assert s.table_dir("country", "HDD") == s.output_dir / "HDD_country"
    assert store_of(s).path("m", "ssp1", WBT_UKMO, ComfortState(22, 60)) == tmp_path / "dd/CDD/m_ssp1_WBT_UKMO_base22C_rh60.nc"
    assert store_of(s, "HDD").path("m", "ssp1", HDD_UKMO, ComfortState(18)) == tmp_path / "heating/HDD/m_ssp1_DBT_UKMO_base18C.nc"
    assert ComfortState(22, 60).csv_name_of("CDD", s.table_name) == "CDD22_60.csv"


def test_command_line_end_to_end(tmp_path, countries, population):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    make_archive(tmp_path / "climate", "historical", [(2008, 2014)])
    make_archive(tmp_path / "climate", "ssp245", [(2015, 2020)], seed=5)
    gpd.GeoDataFrame({"iso_a3": ["AAA", "BBB"]}, geometry=[box(0, 8.5, 1.5, 10.5), box(1.5, 8.5, 2.5, 10.0)],
                     crs=4326).to_file(tmp_path / "land.geojson", driver="GeoJSON")
    countries.save(tmp_path / "n_code.npz", tmp_path / "countries.csv")
    population.save(tmp_path / "pop.npz")
    regions = np.where(countries.codes > 0, 1, 0)
    np.save(tmp_path / "regions.npy", regions)
    (tmp_path / "names.txt").write_text("Toyland\n", encoding="utf-8")
    cfg = tmp_path / "cddh.toml"
    cfg.write_text(f"""
[paths]
climate_dir = "climate"
cdd_dir = "out/cdd"
output_dir = "out"
population = "pop.npz"
land_shapefile = "land.geojson"
country_grid = "n_code.npz"
country_table = "countries.csv"
study_regions = "regions.npy"
study_region_names = "names.txt"
[run]
models = ["{MODEL}"]
scenarios = ["ssp245"]
global_north = [1]
""", encoding="utf-8")
    assert main(["--config", str(cfg), "cdd", "--states", "22_60", "24_40"]) == 0
    files = sorted((tmp_path / "out" / "cdd").rglob("*.nc"))
    assert len(files) == 2 * 4 * 2
    assert main(["--config", str(cfg), "aggregate", "--level", "country", "--states", "22_60"]) == 0
    df = pd.read_csv(tmp_path / "out" / "cdd_country" / "CDD_22C_60p.csv")
    assert set(df.SREGION) == {"AAA", "BBB", "CCC"} and df.YEAR.min() == 2008 and df.YEAR.max() == 2020
    assert main(["--config", str(cfg), "aggregate", "--level", "region", "--states", "24_40"]) == 0
    assert set(pd.read_csv(tmp_path / "out" / "cdd_region" / "CDD_24C_40p.csv").SREGION) == {"R1", "R2"}
    # HDD: one method only, default base 18 C; then the regional HDD table
    assert main(["--config", str(cfg), "hdd", "--methods", "UKMO"]) == 0
    hdd = sorted((tmp_path / "out" / "cdd").rglob("HDD_*.nc"))
    assert [f.name.split(f"_{MODEL}")[0] for f in hdd] == ["HDD_DBT_UKMO", "HDD_DBT_UKMO"]   # historical + ssp245
    assert main(["--config", str(cfg), "aggregate", "--kind", "HDD", "--methods", "UKMO", "--level", "region"]) == 0
    h = pd.read_csv(tmp_path / "out" / "hdd_region" / "HDD_18C.csv")
    assert set(h.CAL) == {"UKMO"} and "HDD" in h.columns and h.YEAR.max() == 2020
    # tables restricted to a year range
    assert main(["--config", str(cfg), "aggregate", "--states", "22_60", "--years", "2010", "2016",
                 "--out", str(tmp_path / "short")]) == 0
    short = pd.read_csv(tmp_path / "short" / "CDD_22C_60p.csv")
    assert (short.YEAR.min(), short.YEAR.max()) == (2010, 2016)
    # decomposition: the periods are an input (no built-in years)
    with pytest.raises(ValueError):
        main(["--config", str(cfg), "decompose", "--states", "22_60"])
    assert main(["--config", str(cfg), "decompose", "--states", "22_60",
                 "--periods", "base=2008-2012", "late=2016-2020"]) == 0
    d = pd.read_csv(tmp_path / "out" / "decomposition" / "decomp_long.csv")
    assert set(d.period) == {"late"} and d.max_abs_err.max() < 1e-9


def test_bundled_zone_data():
    cg = CountryGrid.load(ROOT / "data/country_grid/n_code_05deg.npz", ROOT / "data/country_grid/countries.csv")
    codes = set(np.unique(cg.codes)) - {0}
    assert codes <= set(cg.table.un_code)
    assert cg.table.iso3.is_unique
    regions = cg.table.aim_region.dropna()
    assert regions.nunique() > 50
    z = np.load(ROOT / "data/land_cells/land_cells_05deg.npz")
    assert len(z["lat"]) == len(z["lon"]) == len(z["label"]) == 58841 and z["lat"].min() > -60
    land = LandMask.load(ROOT / "data/land_cells/land_cells_05deg.npz", GRID.__class__().lat, GRID.__class__().lon)
    assert land.n == 58841
    r9 = np.load(ROOT / "data/study_regions/region9_05deg.npy")
    names = (ROOT / "data/study_regions/region9_names.txt").read_text(encoding="utf-8").split("\n")
    assert r9.shape == GRID.__class__().shape and r9.max() == len([n for n in names if n.strip()])


LOCAL_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][A-Za-z_]|/Users/|/home/[a-z]|AppData|OneDrive")


def test_no_local_paths_in_repository():
    try:
        files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    bad = []
    for f in files:
        p = ROOT / f
        if p.suffix in {".npy", ".npz", ".nc"} or not p.is_file() or p == Path(__file__).resolve():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        bad += [f"{f}: {m.group(0)}" for m in LOCAL_PATH.finditer(text)]
    assert not bad, bad
