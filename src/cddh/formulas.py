"""Daily degree-day formulas (cooling and heating) and the Stull (2011) wet-bulb temperature.

Standard CDD : max(T - Tb, 0) with the daily mean temperature.
Standard HDD : max(Tb - T, 0) with the daily mean temperature.
UKMO CDD     : UKMO formula on Tmax, Tmin and Tmean = (Tmax + Tmin) / 2 (the CIBSE TM41 equation)
    Tmax <= Tb                          -> 0
    Tmin >= Tb                          -> Tmean - Tb
    Tmin <  Tb < Tmax, Tmean >  Tb      -> (Tmax - Tb) / 2 - (Tb - Tmin) / 4
    Tmin <  Tb < Tmax, Tmean <= Tb      -> (Tmax - Tb) / 4
UKMO HDD     : the heating counterpart (CIBSE TM41 equation)
    Tmin >= Tb                          -> 0
    Tmax <= Tb                          -> Tb - Tmean
    Tmin <  Tb < Tmax, Tmean <  Tb      -> (Tb - Tmin) / 2 - (Tmax - Tb) / 4
    Tmin <  Tb < Tmax, Tmean >= Tb      -> (Tb - Tmin) / 4
    i.e. hdd_ukmo(Tmax, Tmin, Tb) = cdd_ukmo(-Tmin, -Tmax, -Tb).
WBT          : Stull, R. (2011) Wet-bulb temperature from relative humidity and air temperature.
               J. Appl. Meteor. Climatol. 50, 2267-2269 (RH 5-99 %, T -20 to 50 C, sea-level pressure).

All functions keep the dtype of their inputs (float32 climate data give float32 results), so gridded results do
not depend on whether the computation is done in one piece or in chunks.
"""
import numpy as np


def cdd_standard(temp_c, threshold_c):
    """Daily Standard cooling degree days."""
    return np.maximum(temp_c - threshold_c, 0.0)


def cdd_ukmo(tmax_c, tmin_c, threshold_c):
    """Daily UKMO cooling degree days from daily maximum and minimum temperature (CIBSE TM41 equation)."""
    tmean_c = (tmax_c + tmin_c) / 2.0
    result = np.where(
        tmax_c <= threshold_c,
        0.0,
        np.where(
            tmin_c >= threshold_c,
            tmean_c - threshold_c,
            np.where(tmean_c > threshold_c,
                     (tmax_c - threshold_c) / 2.0 - (threshold_c - tmin_c) / 4.0,
                     (tmax_c - threshold_c) / 4.0),
        ),
    )
    return np.maximum(result, 0.0)


def hdd_standard(temp_c, threshold_c):
    """Daily Standard heating degree days."""
    return np.maximum(threshold_c - temp_c, 0.0)


def hdd_ukmo(tmax_c, tmin_c, threshold_c):
    """Daily UKMO heating degree days from daily maximum and minimum temperature (CIBSE TM41 equation)."""
    tmean_c = (tmax_c + tmin_c) / 2.0
    result = np.where(
        tmin_c >= threshold_c,
        0.0,
        np.where(
            tmax_c <= threshold_c,
            threshold_c - tmean_c,
            np.where(tmean_c < threshold_c,
                     (threshold_c - tmin_c) / 2.0 - (tmax_c - threshold_c) / 4.0,
                     (threshold_c - tmin_c) / 4.0),
        ),
    )
    return np.maximum(result, 0.0)


def wbt_stull_k(temp_k, rh_percent):
    """Wet-bulb temperature (deg C) from air temperature in kelvin (as stored in ISIMIP files) and RH (%)."""
    temp_c = temp_k - 273.15
    return (
        temp_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659))
        + np.arctan(temp_c + rh_percent)
        - np.arctan(rh_percent - 1.676331)
        + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent)
        - 4.686035
    )


def wbt_stull(temp_c, rh_percent):
    """Wet-bulb temperature (deg C) from air temperature (deg C) and RH (%)."""
    return wbt_stull_k(np.asarray(temp_c) + 273.15, rh_percent)


def wbt_threshold(base_temp_c=22.0, base_rh=60.0):
    """WBT base temperature of a dry-bulb comfort state, e.g. 22 C / 60 % -> 16.807 C."""
    return float(wbt_stull_k(base_temp_c + 273.15, base_rh))
