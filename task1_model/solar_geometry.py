# task1_model/solar_geometry.py
import numpy as np
import pandas as pd

def declination_deg(n: int) -> float:
    # δ = 23.45 * sin(360*(284+n)/365)
    return 23.45 * np.sin(np.deg2rad(360.0*(284.0+n)/365.0))

def hour_angle_deg(local_solar_hour: float) -> float:
    # ω = 15*(t-12)
    return 15.0 * (local_solar_hour - 12.0)

def solar_alt_az_deg(lat_deg: float, n: int, local_solar_hour: float) -> tuple[float, float]:
    """
    Simple solar position (no equation of time).
    Returns (altitude_deg, azimuth_deg) in ENU convention:
      azimuth: 0=N, 90=E, 180=S, 270=W
    """
    lat = np.deg2rad(lat_deg)
    dec = np.deg2rad(declination_deg(n))
    ha  = np.deg2rad(hour_angle_deg(local_solar_hour))

    sin_alt = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(ha)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt = np.arcsin(sin_alt)

    # azimuth using cosine form, then quadrant by hour angle
    cos_az = (np.sin(alt)*np.sin(lat) - np.sin(dec)) / (np.cos(alt)*np.cos(lat) + 1e-12)
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az = np.arccos(cos_az)  # 0..pi

    # If afternoon (ha>0), az should be 360-az
    if ha > 0:
        az = 2*np.pi - az

    return float(np.rad2deg(alt)), float(np.rad2deg(az))

def sun_dir_enu(alt_deg: float, az_deg: float) -> np.ndarray:
    """
    Unit vector pointing from point toward the sun in ENU coords:
      x=E, y=N, z=U
    """
    alt = np.deg2rad(alt_deg)
    az  = np.deg2rad(az_deg)
    # horizontal projection magnitude
    ch = np.cos(alt)
    x = ch * np.sin(az)  # East
    y = ch * np.cos(az)  # North
    z = np.sin(alt)      # Up
    v = np.array([x, y, z], float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def facade_dir_irradiance(DNI: float, sun_dir: np.ndarray, facade_az_deg: float) -> float:
    """
    Direct irradiance on a vertical facade (W/m2):
      I_dir,facade = DNI * max(0, dot(sun_dir, n_facade))
    where n_facade is outward horizontal normal.
    """
    az = np.deg2rad(facade_az_deg)
    n_facade = np.array([np.sin(az), np.cos(az), 0.0], float)  # outward
    cos_inc = float(np.dot(sun_dir, n_facade))
    return max(0.0, DNI * cos_inc)

def facade_diffuse_irradiance(DHI: float) -> float:
    """
    Very simple isotropic diffuse: vertical surface gets ~0.5 * DHI.
    You can refine later.
    """
    return 0.5 * max(0.0, DHI)

def within_work_hours(ts: pd.Timestamp, start_h: int, end_h: int) -> bool:
    h = ts.hour
    return (h >= start_h) and (h <= end_h)
