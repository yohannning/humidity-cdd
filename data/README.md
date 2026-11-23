# Bundled data

Only the small files that link grid cells to countries and regions are part of the repository. Climate data,
population rasters and CDD results are not included (see the main README for where to get them).

| File | Content |
|---|---|
| `country_grid/n_code_05deg.npz` | `n_code` (360 x 720, int16): UN M49 country code of each 0.5 deg cell (north-up, row 0 = 89.75 N, column 0 = 179.75 W), built to match the AIM/Hub regional classification; `lat`, `lon` cell centres |
| `country_grid/countries.csv` | `iso3, un_code, country, aim_region`: country table and the AIM/Hub region of each country (102 regions; region `58` = Belgium + Luxembourg; Taiwan = `TWN`) |
| `land_cells/land_cells_05deg.npz` | the 58,841 land cells on which CDD was computed in the paper (`lat`, `lon`, `label` = AIM/Hub region or country of the polygon containing the cell centre; no Antarctica) |
| `land_cells/nearest_land_05deg.npz` | for each populated cell of the country grid (`row`, `col`), the land cell whose CDD it takes in the regional tables (`land_index`; itself for land cells, else the nearest land cell); `land_row`, `land_col` identify the land cells. This is the mapping used for the CDD tables of the paper, saved so that equidistant cells are resolved exactly as there (k-d tree tie-breaking differs between scipy versions) |
| `study_regions/region9_05deg.npy` | nine study regions per 0.5 deg cell (int8, 0 = none, k = line k of `region9_names.txt`), from Natural Earth 1:110m countries with populated coastal cells assigned to the nearest country |
| `study_regions/region9_names.txt` | names of the nine regions (1-4 = Global North) |
| `study_regions/country_region9_naturalearth110m.csv` | country (Natural Earth 1:110m ISO3 or name) -> study region |
