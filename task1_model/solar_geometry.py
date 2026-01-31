# 太阳几何/太阳位置与辐照换算
import numpy as np
import pandas as pd

def declination_deg(n: int) -> float:
    # 赤纬角 δ = 23.45 * sin(360*(284+n)/365)
    return 23.45 * np.sin(np.deg2rad(360.0*(284.0+n)/365.0))

def hour_angle_deg(local_solar_hour: float) -> float:
    # 时角 ω = 15*(t-12)
    return 15.0 * (local_solar_hour - 12.0)

def solar_alt_az_deg(lat_deg: float, n: int, local_solar_hour: float) -> tuple[float, float]:
    """
    Simple solar position (no equation of time).
    Returns (altitude_deg, azimuth_deg) in ENU convention:
      azimuth: 0=N, 90=E, 180=S, 270=W

    简化太阳位置计算（不考虑均时差）。
    返回（高度角 altitude_deg,方位角 azimuth_deg),采用 ENU 坐标约定：
      方位角: 0=北,90=东,180=南,270=西，
    """
    lat = np.deg2rad(lat_deg)
    dec = np.deg2rad(declination_deg(n))
    ha  = np.deg2rad(hour_angle_deg(local_solar_hour))

    sin_alt = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(ha)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt = np.arcsin(sin_alt)

    # 用余弦公式求方位角，再依据时角确定象限
    cos_az = (np.sin(alt)*np.sin(lat) - np.sin(dec)) / (np.cos(alt)*np.cos(lat) + 1e-12)
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az = np.arccos(cos_az)  # 范围 0..π

    # 若为下午（ha>0），方位角应取 360-az
    if ha > 0:
        az = 2*np.pi - az

    return float(np.rad2deg(alt)), float(np.rad2deg(az))

def sun_dir_enu(alt_deg: float, az_deg: float) -> np.ndarray:
    """
    Unit vector pointing from point toward the sun in ENU coords:
      x=E, y=N, z=U

    在 ENU 坐标系下，从观测点指向太阳的单位方向向量：
      x=东,y=北,z=上
    """
    alt = np.deg2rad(alt_deg)
    az  = np.deg2rad(az_deg)
    # 水平投影大小
    ch = np.cos(alt)
    x = ch * np.sin(az)  # East  东向分量
    y = ch * np.cos(az)  # North  北向分量
    z = np.sin(alt)      # Up  竖直向上分量
    v = np.array([x, y, z], float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def facade_dir_irradiance(DNI: float, sun_dir: np.ndarray, facade_az_deg: float) -> float:
    """
    Direct irradiance on a vertical facade (W/m2):
      I_dir,facade = DNI * max(0, dot(sun_dir, n_facade))
    where n_facade is outward horizontal normal.

    立面(竖直面)上的直射辐照度(W/m²):
      I_dir,facade = DNI * max(0, dot(sun_dir, n_facade))
    其中 n_facade 为立面外法线在水平面的方向。
    """
    # 输入：直接辐射DNI（W/m2），sun_dir（(3,) ENU 单位向量），facade_az_deg（度，立面外法线方位角）
    # 输出：立面直射辐照度 I_dir,facade（W/m2）
    az = np.deg2rad(facade_az_deg)
    n_facade = np.array([np.sin(az), np.cos(az), 0.0], float)  # outward  立面外法线（水平）
    cos_inc = float(np.dot(sun_dir, n_facade))
    return max(0.0, DNI * cos_inc)

def facade_diffuse_irradiance(DHI: float) -> float:
    """
    Very simple isotropic diffuse: vertical surface gets ~0.5 * DHI.
    You can refine later.

    极简各向同性散射模型：竖直面获得约 0.5 * DHI 的散射辐照。
    需要时可进一步改进。
    """
    # 输入：散射水平辐射DHI（W/m2，水平面散射辐照度）
    # 输出：立面散射辐照度 I_diff,facade（W/m2），此处近似为 0.5*DHI
    return 0.5 * max(0.0, DHI)

def within_work_hours(ts: pd.Timestamp, start_h: int, end_h: int) -> bool:
    """
    判断时间戳是否落在工作时间段内（包含端点）。
    输入：ts（时间戳）、start_h（开始小时）、end_h（结束小时，包含该小时）。
    输出：若 ts.hour ∈ [start_h, end_h] 返回 True，否则返回 False。
    """
    h = ts.hour
    return (h >= start_h) and (h <= end_h)
