# task1_model/thermal.py  # 中文：Task 1 热负荷/能耗计算模块
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .solar_geometry import within_work_hours

@dataclass(frozen=True)
class EnergyResult:
    E_cool_kWh: float
    E_heat_kWh: float
    Qcool_peak_kW: float
    J: float

def setpoint_temperature(ts: pd.Timestamp, T_cool: float, T_heat: float) -> float:
    # Simple: use cool setpoint if Tout > T_cool else heat setpoint if Tout < T_heat  # 中文：简化：若室外温度高于制冷设定则采用制冷设定；若低于供暖设定则采用供暖设定
    # You can replace with schedule/season logic.  # 中文：可替换为更真实的时间表/季节逻辑
    return (T_cool + T_heat) / 2.0

def internal_gains(ts: pd.Timestamp, work_start: int, work_end: int, Q_work: float, Q_off: float) -> float:
    return Q_work if within_work_hours(ts, work_start, work_end) else Q_off

def compute_energy(
    times: pd.DatetimeIndex,
    Tout: np.ndarray,
    Idir_facade: dict[str, np.ndarray],  # (T,) per facade  # 中文：每个立面的直射辐照时间序列（长度 T）
    Idiff_facade: dict[str, np.ndarray], # (T,) per facade  # 中文：每个立面的散射辐照时间序列（长度 T）
    eta_facade: dict[str, np.ndarray],   # (T,) per facade: lit fraction for direct  # 中文：每个立面的直射“受光比例/未遮挡比例”时间序列（长度 T）
    Awin_facade: dict[str, float],
    UA_total: float,
    SHGC: float,
    k_diff: float,
    hvac_cool_COP: float,
    hvac_heat_eta: float,
    T_in: float,
    work_start: int,
    work_end: int,
    Qint_work: float,
    Qint_off: float,
    dt_hours: float,
    w_cool: float,
    w_heat: float,
) -> EnergyResult:
    """
    Q_load(t) = Q_solar + Q_trans + Q_internal
      Q_solar = SHGC * sum(Awin*(eta*Idir + k_diff*Idiff))
      Q_trans = UA_total * (Tout - Tin)
      Q_internal = schedule
    Positive load -> cooling, negative -> heating

    中文：
    Q_load(t) = Q_solar + Q_trans + Q_internal
      Q_solar = SHGC * Σ[Awin*(eta*Idir + k_diff*Idiff)]（太阳得热）
      Q_trans = UA_total * (Tout - Tin)（围护结构传热）
      Q_internal = 按时间表给定的内部得热
    负荷为正表示需要制冷，负荷为负表示需要供暖。
    """
    T = len(times)
    Q_solar = np.zeros((T,), float)
    for f, A in Awin_facade.items():
        Q_solar += SHGC * A * (eta_facade[f] * Idir_facade[f] + k_diff * Idiff_facade[f])

    Q_trans = UA_total * (Tout - T_in)

    Q_int = np.array([internal_gains(times[k], work_start, work_end, Qint_work, Qint_off) for k in range(T)], float)

    Q_load = Q_solar + Q_trans + Q_int

    Q_cool = np.maximum(Q_load, 0.0)
    Q_heat = np.maximum(-Q_load, 0.0)

    # energy in kWh  # 中文：能耗（单位：kWh）
    E_cool = np.sum((Q_cool / max(1e-9, hvac_cool_COP)) * dt_hours) / 1000.0
    E_heat = np.sum((Q_heat / max(1e-9, hvac_heat_eta)) * dt_hours) / 1000.0

    Qcool_peak_kW = float(np.max(Q_cool) / 1000.0)

    J = w_cool * E_cool + w_heat * E_heat
    return EnergyResult(E_cool_kWh=float(E_cool), E_heat_kWh=float(E_heat), Qcool_peak_kW=Qcool_peak_kW, J=float(J))
