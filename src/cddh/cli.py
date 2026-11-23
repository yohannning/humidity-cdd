"""Command line: cddh {cdd, hdd, population, aggregate, decompose} [options]. Run `cddh <command> -h` for details."""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .aggregate import Aggregator, NearestLand
from .climate import ISIMIP3B_PATTERN, W5E5_PATTERN, ClimateArchive
from .calculator import CDDCalculator
from .config import Settings
from .decomposition import DecompositionAnalysis, parse_periods
from .indicators import INDICATORS, STATES, ComfortState, get_indicator, hdd_indicators
from .land import LandMask
from .population import PopulationGrid
from .store import DEFAULT_TEMPLATE, CDDStore
from .tables import RegionalCDDTable
from .zones import CountryGrid, Zones

PATTERNS = {None: ISIMIP3B_PATTERN, "isimip3b": ISIMIP3B_PATTERN, "w5e5": W5E5_PATTERN}
TEMPLATES = {None: DEFAULT_TEMPLATE, "default": DEFAULT_TEMPLATE}


def store_of(cfg, kind="CDD"):
    """Store of the monthly files of `kind`; HDD uses hdd_dir / hdd_template when set, else the CDD settings."""
    root = cfg.hdd_dir if kind == "HDD" and cfg.hdd_dir is not None else cfg.cdd_dir
    template = cfg.hdd_template if kind == "HDD" and cfg.hdd_template else cfg.cdd_template
    if root is None:
        raise FileNotFoundError("setting 'cdd_dir' (folder of the monthly degree-day files) is not set")
    return CDDStore(root, TEMPLATES.get(template, template))


def states_of(args):
    return [ComfortState.parse(s) for s in args.states] if args.states else list(STATES)


def years_of(args, cfg):
    """(first, last) from --years or the setting `years`; None = all years in the files."""
    if getattr(args, "years", None):
        return tuple(args.years)
    return tuple(cfg.years) if cfg.years else None


def hdd_bases_of(args, cfg):
    return [ComfortState(float(b)) for b in (args.bases or cfg.hdd_bases)]


def hdd_of(args, cfg):
    return list(hdd_indicators(args.methods or cfg.hdd_methods))


def cmd_cdd(args, cfg):
    indicators = [get_indicator(i) for i in args.indicators] if args.indicators else list(INDICATORS)
    run_degree_days(args, cfg, indicators, states_of(args), "CDD")


def cmd_hdd(args, cfg):
    run_degree_days(args, cfg, hdd_of(args, cfg), hdd_bases_of(args, cfg), "HDD")


def run_degree_days(args, cfg, indicators, states, kind):
    archive = ClimateArchive(cfg.require("climate_dir"), PATTERNS.get(cfg.climate_pattern, cfg.climate_pattern))
    store = store_of(cfg, kind)
    for scenario in args.scenarios or [cfg.historical] + cfg.scenarios:
        for model in args.models or cfg.models:
            lat, lon = archive.coords(model, scenario)
            land = (LandMask.from_shapefile(cfg.require("land_shapefile"), lat, lon, exclude=["ATA"])
                    if cfg.land_shapefile else LandMask.load(cfg.require("land_cells"), lat, lon))
            print(f"{model} {scenario}: {land.n} land cells", flush=True)
            years = years_of(args, cfg)
            CDDCalculator(land, indicators, states).run_to_store(
                archive, store, model, scenario, years, overwrite=args.overwrite)


def cmd_population(args, cfg):
    pop = PopulationGrid.from_raster(args.raster)
    out = Path(args.out or cfg.population or Path(cfg.output_dir) / "population_05deg.npy")
    pop.save(out)
    print(f"population {pop.total / 1e9:.3f} billion -> {out.name}")


def regional_aggregator(cfg, level):
    pop = PopulationGrid.load(cfg.require("population"))
    countries = CountryGrid.load(cfg.require("country_grid"), cfg.require("country_table"))
    zones = countries.zones("iso3" if level == "country" else "aim_region")
    nearest = NearestLand.load(cfg.nearest_land) if cfg.nearest_land and Path(cfg.nearest_land).exists() else None
    return Aggregator(pop, zones, fill_offgrid=True, min_value=3.0, nearest=nearest)


def _build_table(job):
    cfg, level, indicators, state, out, years = job
    table = RegionalCDDTable(store_of(cfg, indicators[0].kind), regional_aggregator(cfg, level), cfg.models,
                             cfg.scenarios, cfg.historical, indicators, years=years)
    return str(table.write(state, out, cfg.table_name).name)


def cmd_aggregate(args, cfg):
    if args.kind == "HDD":
        indicators = hdd_of(args, cfg)
        states = [ComfortState.parse(s) for s in args.states] if args.states else hdd_bases_of(args, cfg)
    else:
        indicators, states = list(INDICATORS), states_of(args)
    out = Path(args.out) if args.out else cfg.table_dir(args.level, args.kind)
    jobs = [(cfg, args.level, indicators, st, out, years_of(args, cfg)) for st in states]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for name in ex.map(_build_table, jobs):
            print(f"wrote {name}", flush=True)


def study_zones(cfg, grid):
    names = Path(cfg.require("study_region_names")).read_text(encoding="utf-8").split("\n")
    names = [n.strip() for n in names if n.strip()]
    north = [names[k - 1] for k in cfg.global_north]
    return Zones.from_codes(np.load(cfg.require("study_regions")), names, grid, composites={
        "Global North": north, "Global South": [n for n in names if n not in north], "World": "*"})


def cmd_decompose(args, cfg):
    pop = PopulationGrid.load(cfg.require("population"))
    zones = study_zones(cfg, pop.grid) if args.zones == "study" else CountryGrid.load(
        cfg.require("country_grid"), cfg.require("country_table")).zones(
        "iso3" if args.zones == "country" else "aim_region", {"World": "*"})
    if not (args.periods or cfg.decomposition_periods):
        raise ValueError("set the periods to compare: --periods base=FIRST-LAST NAME=FIRST-LAST ... "
                         "or the setting decomposition_periods")
    periods = parse_periods(args.periods or cfg.decomposition_periods)
    analysis = DecompositionAnalysis(store_of(cfg), Aggregator(pop, zones, fill_offgrid=False), periods,
                                     historical=cfg.historical)
    df = analysis.run(cfg.models, cfg.scenarios, states_of(args))
    out = Path(args.out or cfg.decomposition_dir or Path(cfg.output_dir) / "decomposition")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "decomp_long.csv", index=False)
    analysis.summarize(df).to_csv(out / "decomp_summary.csv", index=False)
    print(f"{len(df)} rows, max closure error {df.max_abs_err.max():.1e} -> {out.name}/decomp_long.csv")


def parser():
    p = argparse.ArgumentParser(prog="cddh", description=__doc__)
    p.add_argument("--config", help="settings file (default: $CDDH_CONFIG or ./cddh.toml)")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cdd", help="monthly land-cell CDD from daily climate files")
    c.add_argument("--models", nargs="+")
    c.add_argument("--scenarios", nargs="+", help="default: historical + configured scenarios")
    c.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    c.add_argument("--indicators", nargs="+", help="e.g. DBT-Standard WBT-UKMO (default: all four)")
    c.add_argument("--states", nargs="+", help="comfort states T_RH, e.g. 22_60 (default: all eight)")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_cdd)

    c = sub.add_parser("hdd", help="monthly land-cell HDD (dry-bulb temperature) from daily climate files")
    c.add_argument("--models", nargs="+")
    c.add_argument("--scenarios", nargs="+", help="default: historical + configured scenarios")
    c.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    c.add_argument("--methods", nargs="+", choices=["Standard", "UKMO"],
                   help="HDD method(s) (default: setting hdd_methods, i.e. both)")
    c.add_argument("--bases", nargs="+", type=float, help="base temperature(s) in C (default: setting hdd_bases, 18)")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_hdd)

    c = sub.add_parser("population", help="sum a fine population raster to the 0.5 deg grid")
    c.add_argument("--raster", required=True)
    c.add_argument("--out")
    c.set_defaults(func=cmd_population)

    c = sub.add_parser("aggregate", help="population-weighted CDD or HDD per country (ISO3) or AIM/Hub region")
    c.add_argument("--kind", choices=["CDD", "HDD"], default="CDD")
    c.add_argument("--level", choices=["country", "region"], default="region")
    c.add_argument("--states", nargs="+", help="CDD comfort states T_RH (e.g. 22_60) or HDD bases T (e.g. 18)")
    c.add_argument("--methods", nargs="+", choices=["Standard", "UKMO"], help="HDD only: method(s) to include")
    c.add_argument("--bases", nargs="+", type=float, help="HDD only: base temperature(s) in C")
    c.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                   help="year range of the tables (default: setting years, else all years in the files)")
    c.add_argument("--out")
    c.add_argument("--jobs", type=int, default=1)
    c.set_defaults(func=cmd_aggregate)

    c = sub.add_parser("decompose", help="growth-factor gap decomposition between indicator pairs")
    c.add_argument("--zones", choices=["study", "region", "country"], default="study")
    c.add_argument("--periods", nargs="+", metavar="NAME=FIRST-LAST",
                   help="baseline 'base=FIRST-LAST' and target periods (default: setting decomposition_periods)")
    c.add_argument("--states", nargs="+")
    c.add_argument("--out")
    c.set_defaults(func=cmd_decompose)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = Settings.load(args.config)
    args.func(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
