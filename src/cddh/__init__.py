"""Humidity-aware cooling degree days (and dry-bulb heating degree days): gridded CDD/HDD from daily climate data,
population-weighted aggregation to countries (ISO3) and AIM/Hub regions, and the growth-factor decomposition between CDD
indicators."""
from .aggregate import Aggregator, NearestLand, weighted_mean_by_group
from .calculator import CDDCalculator
from .climate import ISIMIP3B_PATTERN, W5E5_PATTERN, ClimateArchive, DailyFields
from .config import Settings
from .decomposition import DecompositionAnalysis, GrowthFactorGap
from .grid import GLOBAL_05, Grid
from .indicators import (DBT_STANDARD, DBT_UKMO, HDD_BASE, HDD_BASES, HDD_INDICATORS, HDD_STANDARD, HDD_UKMO,
                         INDICATORS, MAIN_STATE, STATES, WBT_STANDARD, WBT_UKMO, ComfortState, Indicator,
                         get_indicator, hdd_indicators)
from .land import LandMask
from .population import PopulationGrid
from .store import DEFAULT_TEMPLATE, AnnualCDD, CDDStore
from .tables import RegionalCDDTable
from .zones import CountryGrid, Zones

__version__ = "1.0.0"
