# cddh: humidity-aware cooling degree days

Code for the cooling-degree-day (CDD) part of *Humidity reshapes projected growth in cooling demand and its economic
impacts*:

1. **CDD (and HDD) from daily climate data**: degree days are computed for every day and 0.5° land cell from daily
   NetCDF files (ISIMIP3b bias-adjusted GCM output or GSWP3-W5E5) and summed to **monthly** totals, which are saved
   per land cell. CDD has four indicators and eight comfort states; HDD uses dry-bulb temperature with the Standard or
   UKMO method (base 18 °C by default).
2. **Population weighting**: a fine population raster (e.g. WorldPop, 1 km) is summed onto the 0.5° grid.
3. **Aggregation**: the monthly totals are summed to **annual** degree days per land cell (complete years only) and
   averaged with population weights per country (ISO3) and per AIM/Hub region. The output is one row per year in the
   AIM/Hub degree-day input format.
4. **Decomposition**: growth factors of the indicators between a baseline period and target periods of your choice,
   and the decomposition of the growth-factor gap between indicator pairs into an increment effect and a baseline
   effect.

```
daily T, RH ──► daily degree days ──► monthly sum per land cell ──► annual sum per land cell ──► population-weighted
(NetCDF)        (formulas.py)         (cddh cdd/hdd, NetCDF)        (complete years)            annual table (cddh aggregate)
```

The repository holds no climate, population or CDD data, only the small files that link grid cells to countries and
regions (`data/`, see `data/README.md`). The tests run on synthetic data.

## Indicators and comfort states

| Indicator | Daily CDD |
|---|---|
| DBT-Standard | max(T_mean − T_b, 0) |
| DBT-UKMO | UKMO formula on T_max, T_min |
| WBT-Standard | max(WBT(T_mean, RH) − T_b,wb, 0) |
| WBT-UKMO | UKMO formula on WBT(T_max, RH), WBT(T_min, RH) |

- **WBT:** Stull (2011) wet-bulb temperature from air temperature and daily mean relative humidity.
- **Comfort states:** dry-bulb base temperature T_b = 20, 22, 24 or 26 °C, combined with RH 40 % or 60 %.
- **WBT base:** T_b,wb = WBT(T_b, RH), e.g. 22 °C / 60 % → 16.807 °C. DBT indicators do not depend on RH.
- **Main state** of the paper: 22 °C / 60 %.
- **UKMO formula:** `cddh.formulas.cdd_ukmo` (the CIBSE TM41 equation):

  | Case | Daily CDD |
  |---|---|
  | T_max ≤ T_b | 0 |
  | T_min ≥ T_b | T̄ − T_b |
  | T_min < T_b < T_max, T̄ > T_b | (T_max − T_b)/2 − (T_b − T_min)/4 |
  | T_min < T_b < T_max, T̄ ≤ T_b | (T_max − T_b)/4 |

  T̄ = (T_max + T_min)/2.

### Heating degree days (HDD)

HDD uses dry-bulb temperature only; choose the Standard method, the UKMO method, or both. The base temperature has
no humidity (default 18 °C, setting `hdd_bases`).

| Indicator | Daily HDD |
|---|---|
| HDD-DBT-Standard | max(T_b − T_mean, 0) |
| HDD-DBT-UKMO | UKMO formula on T_max, T_min (`cddh.formulas.hdd_ukmo`, the CIBSE TM41 equation) |

| Case | Daily HDD |
|---|---|
| T_min ≥ T_b | 0 |
| T_max ≤ T_b | T_b − T̄ |
| T_min < T_b < T_max, T̄ < T_b | (T_b − T_min)/2 − (T_max − T_b)/4 |
| T_min < T_b < T_max, T̄ ≥ T_b | (T_b − T_min)/4 |

## Installation

```bash
conda env create -f environment.yml
conda activate cddh
pip install -e .
pytest
```

Required: numpy, pandas, scipy and netCDF4. Optional:
- geopandas, to build land cells from a country shapefile;
- rasterio, to read the WorldPop GeoTIFF.

## Settings

No file location is written in the code. Copy `config/cddh.example.toml` to `cddh.toml` (ignored by git) and fill in
the paths on your machine. Alternatively:
- set environment variables `CDDH_<KEY>`, e.g. `CDDH_CLIMATE_DIR`;
- or pass `--config <file>`.

| Key | Content |
|---|---|
| `climate_dir` | daily climate archive; ISIMIP3b layout `<model>/<scenario>/<var>/<model>_<member>_w5e5_<scenario>_<var>_global_daily_<y0>_<y1>.nc` (`climate_pattern = "w5e5"` for GSWP3-W5E5 obsclim) |
| `cdd_dir`, `cdd_template` | monthly land-cell degree-day files (output of `cddh cdd` / `cddh hdd`, input of `aggregate` and `decompose`) and their file name template, default `{model}/{scenario}/{kind}_{temperature}_{method}_{model}_{scenario}_{tag}_monthly.nc` |
| `hdd_dir`, `hdd_template` | optional: a different folder / template for HDD (default: the CDD settings) |
| `region_table_dir`, `country_table_dir`, `table_name` | where the tables go (default `<output_dir>/{kind}_region`, `<output_dir>/{kind}_country`) and their name (default `{kind}_{state}.csv`, e.g. `CDD_22C_60p.csv`) |
| `decomposition_dir` | decomposition output (default `<output_dir>/decomposition`) |
| `population` | 0.5° population grid (output of `cddh population`) |
| `land_cells` | land cells (default: the bundled cells of the paper) |
| `nearest_land` | populated cell → land cell mapping for the tables (default: bundled) |
| `land_shapefile` | optional: country polygons used instead of `land_cells` |
| `country_grid`, `country_table`, `study_regions`, `study_region_names` | default: bundled files in `data/` |
| `hdd_bases`, `hdd_methods` | HDD base temperatures in °C (default `[18]`) and methods (default `["Standard", "UKMO"]`) |
| `years` | optional `[first, last]` year range to compute and tabulate (default: every year in the climate files) |
| `decomposition_periods` | periods compared by `cddh decompose`, e.g. `{ base = [2010, 2019], end = [2090, 2099] }`; `base` is the baseline (no default) |
| `models`, `scenarios`, `historical` | what to run (default: GFDL-ESM4, IPSL-CM6A-LR, MPI-ESM1-2-HR, UKESM1-0-LL; SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5) |

## Usage

```bash
# 1. daily CDD summed to monthly totals on land cells: historical + scenarios, 4 indicators x 8 comfort states
cddh cdd                                   # or: --models gfdl-esm4 --scenarios ssp245 --states 22_60 --years 2015 2050
# 2. population weights: fine population raster -> 0.5 deg
cddh population --raster <population raster, e.g. WorldPop 2020 1 km GeoTIFF>
# 3. annual, population-weighted CDD per AIM/Hub region or per country (ISO3), one CSV per comfort state
cddh aggregate --level region --jobs 4     # results/CDD_region/CDD_22C_60p.csv, ... (optional: --years FIRST LAST)
cddh aggregate --level country
# HDD (dry-bulb): Standard and/or UKMO, base 18 C unless --bases is given
cddh hdd --methods UKMO                    # or: --methods Standard UKMO --bases 18 15.5
cddh aggregate --kind HDD --methods UKMO   # results/HDD_region/HDD_18C.csv
# 4. growth-factor decomposition (nine study regions, Global North/South, World) between periods of your choice
cddh decompose --periods base=2010-2019 mid=2050-2059 end=2090-2099
                                           # results/decomposition/decomp_long.csv, decomp_summary.csv
```

`python -m cddh ...` works as well.

### Outputs

- **`cddh cdd`** writes one NetCDF file per model, scenario, indicator and comfort state, at `<cdd_dir>/<cdd_template>`
  (default `<model>/<scenario>/CDD_<DBT|WBT>_<Standard|UKMO>_<model>_<scenario>_base<T>C_rh<RH>_monthly.nc`).
  - Contents: `CDD(time, grid)`, the **monthly sum** of daily degree days of each land cell, plus the `lat`, `lon`
    and label of each land cell. Daily values are not stored.
- **`cddh hdd`** writes the same layout with `kind = HDD` (default `<model>/<scenario>/HDD_DBT_<Standard|UKMO>_<model>_<scenario>_base<T>C_monthly.nc`)
  with the variable `HDD(time, grid)`.
- **`cddh aggregate`** writes `CDD_<T>C_<RH>p.csv` (`HDD_<T>C.csv` with `--kind HDD`, last column `HDD`) with the columns `SREGION, GCM, SSP, RCP, TEMP, CAL, YEAR, CDD`.
  - Each row is a year: the 12 monthly totals of a land cell are summed (only complete years are used), then
    averaged over the cells of a country or region.
  - Years: those of the historical run followed by those of each scenario, i.e. every year in the files, or the
    range given by `years` / `--years`.
  - The weighted mean is CDD = Σ pop_i·CDD_i / Σ pop_i over all populated cells of a country or region.
  - Populated cells that are not land cells (coasts, small islands) take the CDD of the nearest land cell, using the
    bundled mapping `data/land_cells/nearest_land_05deg.npz` (the one used for the tables of the paper, so they are
    reproduced byte for byte). Without it, ties between equidistant land cells go to the smaller land index.
  - Values below 3 degree days are set to 0, as in the AIM/Hub input.
- **`cddh decompose`** compares the mean annual degree days of the `base` period with each target period
  (`--periods` or the setting `decomposition_periods`; there is no default).
  - It uses population-weighted means over the land cells of each zone.
  - Identity: GF_A − GF_B = (ΔA − ΔB)/A₀ [increment] + ΔB(1/A₀ − 1/B₀) [baseline].
  - The symmetric form is reported too. Every row is checked to close to 1e-9.

## Code structure

```
src/cddh/
  formulas.py       daily CDD and HDD (Standard, UKMO) and Stull WBT
  indicators.py     Indicator (CDD: DBT/WBT x Standard/UKMO; HDD: DBT x Standard/UKMO), ComfortState (T_b, RH)
  grid.py           Grid (0.5 deg cell-centre grid), block sums, ESRI ASCII reader
  climate.py        ClimateArchive (find daily files), DailyFields (read whole months on land cells)
  land.py           LandMask (land cells from the bundled list or a shapefile)
  calculator.py     CDDCalculator (daily degree days -> monthly sums; all indicators x states in one pass; CDD or HDD)
  store.py          CDDStore (write/read monthly land-cell files; annual sums), AnnualCDD
  population.py     PopulationGrid (WorldPop raster -> 0.5 deg, load/save)
  zones.py          CountryGrid (cell -> UN code -> ISO3 / AIM region), Zones (labels + composite groups)
  aggregate.py      Aggregator (population-weighted means, nearest-land fill, minimum value)
  tables.py         RegionalCDDTable (AIM/Hub CSV format; CDD or HDD)
  decomposition.py  GrowthFactorGap, DecompositionAnalysis (user-defined base and target periods)
  config.py         Settings (TOML file + environment variables)
  cli.py            command line
tests/              synthetic-data tests (no real data needed)
data/               country, region and land-cell grids (see data/README.md)
```

Example in Python:

```python
from cddh import (WBT_STANDARD, Aggregator, CDDCalculator, CDDStore, ClimateArchive, ComfortState, CountryGrid,
                  LandMask, PopulationGrid, Settings)

cfg = Settings.load("cddh.toml")
archive = ClimateArchive(cfg.climate_dir)
lat, lon = archive.coords("gfdl-esm4", "ssp245")
land = LandMask.load(cfg.land_cells, lat, lon)
months, cdd = CDDCalculator(land, states=[ComfortState(22, 60)]).run(archive, "gfdl-esm4", "ssp245", (2015, 2020))

countries = CountryGrid.load(cfg.country_grid, cfg.country_table)
agg = Aggregator(PopulationGrid.load(cfg.population), countries.zones("iso3"), min_value=3.0)
annual = CDDStore(cfg.cdd_dir).read_series("gfdl-esm4", ["historical", "ssp245"], WBT_STANDARD, ComfortState(22, 60))
by_country = agg.means(annual.values, annual.lat, annual.lon)     # {ISO3: annual CDD, one value per year}
```

## Input data (not included)

| Data | Source |
|---|---|
| ISIMIP3b bias-adjusted daily climate (`tas`, `tasmax`, `tasmin`, `hurs`) | ISIMIP repository, https://data.isimip.org (CC BY 4.0) |
| GSWP3-W5E5 obsclim (optional, observation-based check) | ISIMIP3a, https://data.isimip.org (CC BY 4.0) |
| WorldPop 2020 global 1 km population (constrained, UN-adjusted) | https://www.worldpop.org (CC BY 4.0) |
| Natural Earth 1:110m admin-0 countries (optional) | https://www.naturalearthdata.com (public domain) |

## Licence

MIT (see `LICENSE`). The bundled country grid was built to match the AIM/Hub regional classification, and the study-region grid is based on Natural Earth.
