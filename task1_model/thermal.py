# task1_model/thermal.py
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
    # Simple: use cool setpoint if Tout > T_cool else heat setpoint if Tout < T_heat
    # You can replace with schedule/season logic.
    return (T_cool + T_heat) / 2.0

def internal_gains(ts: pd.Timestamp, work_start: int, work_end: int, Q_work: float, Q_off: float) -> float:
    return Q_work if within_work_hours(ts, work_start, work_end) else Q_off

def compute_energy(
    times: pd.DatetimeIndex,
    Tout: np.ndarray,
    Idir_facade: dict[str, np.ndarray],  # (T,) per facade
    Idiff_facade: dict[str, np.ndarray], # (T,) per facade
    eta_facade: dict[str, np.ndarray],   # (T,) per facade: lit fraction for direct
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

    # energy in kWh
    E_cool = np.sum((Q_cool / max(1e-9, hvac_cool_COP)) * dt_hours) / 1000.0
    E_heat = np.sum((Q_heat / max(1e-9, hvac_heat_eta)) * dt_hours) / 1000.0

    Qcool_peak_kW = float(np.max(Q_cool) / 1000.0)

    J = w_cool * E_cool + w_heat * E_heat
    return EnergyResult(E_cool_kWh=float(E_cool), E_heat_kWh=float(E_heat), Qcool_peak_kW=Qcool_peak_kW, J=float(J))
