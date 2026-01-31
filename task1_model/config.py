# task1_model/config.py  Task 1 参数配置（建筑/围护结构/HVAC/内部得热/采光/舒适性）
from dataclasses import dataclass

@dataclass(frozen=True)
class BuildingGeom:
    L: float = 60.0   # m  建筑长度（米）
    W: float = 24.0   # m  建筑宽度（米）
    H: float = 8.0    # m total (2 floors)  建筑总高度（米，含 2 层）
    wwr_south: float = 0.45
    wwr_other: float = 0.30

@dataclass(frozen=True)
class HVACParams:
    # Setpoints (degC)  温度设定值（℃）
    T_cool: float = 24.0
    T_heat: float = 20.0
    # Efficiencies  效率/性能系数
    COP_cool: float = 3.2
    eta_heat: float = 0.9

@dataclass(frozen=True)
class EnvelopeParams:
    # 简化参数，可在后续细化
    U_wall: float = 0.6   # 墙体传热系数 U（W/m²·K）
    U_win: float = 2.4    # 窗体传热系数 U（W/m²·K）
    SHGC: float = 0.6     # 太阳得热系数（基准值，可在优化中调整）
    tau_diff: float = 0.6 # 散射光可见光透射率近似值

@dataclass(frozen=True)
class InternalGains:
    # Very simple schedule: W  简化的内部得热时间表（单位：W）
    # You can replace with q_person*N_occ + q_equip  # 中文：可替换为“人员散热 + 设备散热”等更真实模型
    Q_internal_work: float = 18000.0  # W (example)  # 中文：工作时内部得热（示例）
    Q_internal_off: float = 3000.0    # W (example)  # 中文：非工作时内部得热（示例）

@dataclass(frozen=True)
class DaylightingParams:
    # Diffuse-only daylight model: E_in = C_dl * kappa * DHI * sum(tau*A*kdiff) / A_floor  # 中文：仅散射光采光模型（见公式）
    kappa: float = 120.0  # lux per (W/m2) rough mapping for diffuse (tunable)  # 中文：散射辐照到照度的粗略换算系数（可调）
    C_dl: float = 0.1   # transmission-to-average factor (tunable)  # 中文：透射到平均照度的系数（可调）
    C_dir: float = 0.05  # direct-to-average factor (tunable)  # 中文：直射贡献到平均照度的系数（可调）
    k_diff_shade: float = 0.9  # shading blocks diffuse weakly  # 中文：遮阳对散射光的削弱系数（弱削弱）

@dataclass(frozen=True)
class ComfortConstraints:
    glare_depth_m: float = 3
    glare_hours_max: float = 1000.0
    daylight_lux_min: float = 250.0
    daylight_ok_hours_min: float = 1500.0  # work-hour timesteps with E_in >= daylight_lux_min  # 中文：工作时段内“照度合格(>=阈值)”的全年最少小时数
    work_start_hour: int = 8
    work_end_hour: int = 17
@dataclass(frozen=True)
class OptimizationWeights:
    w_cool: float = 1.0
    w_heat: float = 1.0
