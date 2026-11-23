"""Degree-day indicators (CDD/HDD x temperature index x method) and comfort states (base temperature x RH).

CDD: DBT or WBT temperature index, Standard or UKMO method, base = comfort state (temperature and RH).
HDD: DBT only, Standard or UKMO method, base = temperature only (default 18 C).
"""
from dataclasses import dataclass

from .formulas import cdd_standard, cdd_ukmo, hdd_standard, hdd_ukmo, wbt_stull_k, wbt_threshold

KINDS = ("CDD", "HDD")
METHODS = ("Standard", "UKMO")


@dataclass(frozen=True)
class ComfortState:
    """Indoor comfort state: dry-bulb base temperature (C) and relative humidity (%).

    rh_pct is None for a base without humidity (HDD, which uses dry-bulb temperature only).
    """

    temp_c: float
    rh_pct: float = None

    @classmethod
    def parse(cls, text):
        """'22_60' -> ComfortState(22, 60); '18' -> ComfortState(18) (no RH)."""
        parts = str(text).split("_")
        if len(parts) == 1:
            return cls(float(parts[0]))
        t, rh = parts
        return cls(float(t), float(rh))

    @property
    def label(self):
        return f"{self.temp_c:g}" if self.rh_pct is None else f"{self.temp_c:g}_{self.rh_pct:g}"

    @property
    def tag(self):
        """File tag, e.g. 'base22C_rh60' or 'base18C'."""
        return f"base{self.temp_c:g}C" + ("" if self.rh_pct is None else f"_rh{self.rh_pct:g}")

    @property
    def short(self):
        """Short label used in table names, e.g. '22C_60p' or '18C'."""
        return f"{self.temp_c:g}C" + ("" if self.rh_pct is None else f"_{self.rh_pct:g}p")

    def csv_name_of(self, kind="CDD", template="{kind}_{state}.csv"):
        """Name of the regional table of this state, e.g. 'CDD_22C_60p.csv' or 'HDD_18C.csv'."""
        return template.format(kind=kind, state=self.short, temp=f"{self.temp_c:g}",
                               rh="" if self.rh_pct is None else f"{self.rh_pct:g}")

    @property
    def csv_name(self):
        return self.csv_name_of("CDD")

    def __str__(self):
        return f"{self.temp_c:g} C" + ("" if self.rh_pct is None else f" / {self.rh_pct:g} %")


MAIN_STATE = ComfortState(22.0, 60.0)
STATES = tuple(ComfortState(t, rh) for t in (20.0, 22.0, 24.0, 26.0) for rh in (40.0, 60.0))
HDD_BASE = ComfortState(18.0)
HDD_BASES = (HDD_BASE,)


def temperature(index, fields, var):
    """Daily temperature (C) of a temperature index ('DBT' or 'WBT') from fields {variable: K or %}."""
    if index == "DBT":
        return fields[var] - 273.15
    return wbt_stull_k(fields[var], fields["hurs"])


@dataclass(frozen=True)
class Indicator:
    """A degree-day indicator.

    temperature : 'DBT' (dry-bulb air temperature) or 'WBT' (Stull wet-bulb temperature from T and RH)
    method      : 'Standard' (daily mean) or 'UKMO' (daily maximum and minimum; CIBSE TM41 equation)
    kind        : 'CDD' (cooling) or 'HDD' (heating; DBT only)
    """

    temperature: str
    method: str
    kind: str = "CDD"

    def __post_init__(self):
        if self.kind not in KINDS or self.method not in METHODS or self.temperature not in ("DBT", "WBT"):
            raise ValueError(f"unknown indicator {self.kind} {self.temperature} {self.method}")
        if self.kind == "HDD" and self.temperature != "DBT":
            raise ValueError("HDD is defined for dry-bulb temperature (DBT) only")

    @property
    def name(self):
        """'DBT-Standard' for CDD (as in the published tables), 'HDD-DBT-Standard' for HDD."""
        base = f"{self.temperature}-{self.method}"
        return base if self.kind == "CDD" else f"{self.kind}-{base}"

    @property
    def key(self):
        base = f"{self.temperature}_{self.method}"
        return base if self.kind == "CDD" else f"{self.kind}_{base}"

    @property
    def temperature_variables(self):
        return ("tas",) if self.method == "Standard" else ("tasmax", "tasmin")

    @property
    def variables(self):
        """Climate variables needed to compute the indicator."""
        return self.temperature_variables + (("hurs",) if self.temperature == "WBT" else ())

    def threshold(self, state):
        """Base temperature (C) of this indicator for a comfort state."""
        if self.temperature == "DBT":
            return state.temp_c
        if state.rh_pct is None:
            raise ValueError(f"{self.name} needs a comfort state with relative humidity")
        return wbt_threshold(state.temp_c, state.rh_pct)

    def depends_on_rh(self):
        return self.temperature == "WBT"

    def temperatures(self, fields):
        """Temperature (C) used by the indicator from daily fields {variable: array in K or %}."""
        return {v: temperature(self.temperature, fields, v) for v in self.temperature_variables}

    def daily(self, temps, threshold_c):
        """Daily degree days from the temperatures returned by `temperatures`."""
        if self.kind == "CDD":
            if self.method == "Standard":
                return cdd_standard(temps["tas"], threshold_c)
            return cdd_ukmo(temps["tasmax"], temps["tasmin"], threshold_c)
        if self.method == "Standard":
            return hdd_standard(temps["tas"], threshold_c)
        return hdd_ukmo(temps["tasmax"], temps["tasmin"], threshold_c)

    def __str__(self):
        return self.name


DBT_STANDARD = Indicator("DBT", "Standard")
DBT_UKMO = Indicator("DBT", "UKMO")
WBT_STANDARD = Indicator("WBT", "Standard")
WBT_UKMO = Indicator("WBT", "UKMO")
INDICATORS = (DBT_STANDARD, DBT_UKMO, WBT_STANDARD, WBT_UKMO)

HDD_STANDARD = Indicator("DBT", "Standard", "HDD")
HDD_UKMO = Indicator("DBT", "UKMO", "HDD")
HDD_INDICATORS = (HDD_STANDARD, HDD_UKMO)


def get_indicator(name):
    """Indicator by name ('DBT-Standard', 'HDD-DBT-UKMO') or key ('DBT_Standard', 'HDD_DBT_UKMO')."""
    for ind in INDICATORS + HDD_INDICATORS:
        if name in (ind.name, ind.key):
            return ind
    raise KeyError(name)


def hdd_indicators(methods=METHODS):
    """HDD indicators of the chosen methods ('Standard', 'UKMO'; case-insensitive)."""
    wanted = {m.lower() for m in methods}
    unknown = wanted - {m.lower() for m in METHODS}
    if unknown:
        raise KeyError(f"unknown HDD method(s): {sorted(unknown)}")
    return tuple(i for i in HDD_INDICATORS if i.method.lower() in wanted)
