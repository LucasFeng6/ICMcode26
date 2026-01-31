# task1_model/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class BuildingGeom:
    L: float = 60.0   # m
    W: float = 24.0   # m
    H: float = 8.0    # m total (2 floors)
    wwr_south: float = 0.45
    wwr_other: float = 0.30

@dataclass(frozen=True)
class HVACParams:
    # Setpoints (degC)
    T_cool: float = 24.0
    T_heat: float = 20.0
    # Efficiencies
    COP_cool: float = 3.2
    eta_heat: float = 0.9

@dataclass(frozen=True)
class EnvelopeParams:
    # Simplified, can be refined later
    U_wall: float = 0.6   # W/m2K
    U_win: float = 2.4    # W/m2K
    SHGC: float = 0.6     # baseline; you can optimize later
    tau_diff: float = 0.6 # visible transmittance approx for diffuse

@dataclass(frozen=True)
class InternalGains:
    # Very simple schedule: W
    # You can replace with q_person*N_occ + q_equip
    Q_internal_work: float = 18000.0  # W (example)
    Q_internal_off: float = 3000.0    # W (example)

@dataclass(frozen=True)
class DaylightingParams:
    # Diffuse-only daylight model: E_in = C_dl * kappa * DHI * sum(tau*A*kdiff) / A_floor
    kappa: float = 120.0  # lux per (W/m2) rough mapping for diffuse (tunable)
    C_dl: float = 0.012   # transmission-to-average factor (tunable)
    k_diff_shade: float = 0.9  # shading blocks diffuse weakly

@dataclass(frozen=True)
class ComfortConstraints:
    glare_depth_m: float = 1.5
    glare_hours_max: float = 50.0
    daylight_lux_min: float = 300.0
    work_start_hour: int = 8
    work_end_hour: int = 17  # inclusive of 17:00 hour start; adjust in code if needed

@dataclass(frozen=True)
class OptimizationWeights:
    w_cool: float = 1.0
    w_heat: float = 1.0
