# task1_model/evaluate.py  # 中文：方案评估（遮阳、眩光、采光、能耗与指标）
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .frames import Frame
from .sampling import WindowSpec, rect_grid_samples_local
from .shading_devices import OverhangParams, FinParams, make_overhang_mesh_local, make_fins_mesh_local
from .ray_scene import RayScene
from .solar_geometry import solar_alt_az_deg, sun_dir_enu, facade_dir_irradiance, facade_diffuse_irradiance
from .comfort import compute_glare_hours, daylight_ok
from .thermal import compute_energy, EnergyResult

@dataclass(frozen=True)
class Design:
    D_oh: float
    D_fin: float
    beta_fin_deg: float

@dataclass(frozen=True)
class EvalOutputs:
    feasible: bool
    glare_hours: float
    daylight_ok: bool
    energy: EnergyResult
    AESR: float
    PLR: float

def build_facade_scene_and_samples(
    facade: str,
    frame: Frame,
    win: WindowSpec,
    nx: int,
    ny: int,
    design: Design,
) -> tuple[RayScene, np.ndarray]:
    """
    Create scene (overhang + fins) in LOCAL coords and window samples in LOCAL coords.

    中文：
    在局部坐标系中创建遮阳场景（挑檐 + 侧翼），并在局部坐标系中生成窗面采样点。
    """
    pts_local = rect_grid_samples_local(win, nx=nx, ny=ny)
    win_bottom = win.sill_h
    win_top = win.sill_h + win.height

    if facade.upper() == "S":
        # South facade: overhang only
        over = make_overhang_mesh_local(
            win_width=win.width,
            win_top_v=win_top,
            p=OverhangParams(depth=design.D_oh),
        )
        fins = None
    else:
        # Non-south facades: fins only
        over = None
        fins = make_fins_mesh_local(
            win_width=win.width,
            win_bottom_v=win_bottom,
            win_top_v=win_top,
            p=FinParams(depth=design.D_fin, angle_deg=design.beta_fin_deg),
        )

    scene = RayScene.from_meshes([over, fins])
    return scene, pts_local

def compute_eta_series(
    times: pd.DatetimeIndex,
    lat_deg: float,
    frame: Frame,
    scene: RayScene,
    sample_pts_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For a single facade: compute
      sun_dir_world (T,3)
      eta(t) = lit fraction for direct (unshaded fraction among all samples)
    NOTE: We treat samples with sun behind facade as eta=0 (no direct).

    中文：
    针对单个立面，计算：
      sun_dir_world（T,3）的太阳方向（世界坐标）
      eta(t)：直射光“受光比例/未遮挡比例”（在所有采样点中未被遮挡的比例）
    注意：若太阳在立面背后，则视为 eta=0（无直射）。
    """
    T = len(times)
    eta = np.zeros((T,), float)
    sun_dirs = np.zeros((T, 3), float)

    for k, ts in enumerate(times):
        n = int(ts.dayofyear)
        local_hour = ts.hour + ts.minute/60.0
        alt, az = solar_alt_az_deg(lat_deg, n, local_hour)
        if alt <= 0.0:
            sun_dirs[k] = np.array([0.0, 0.0, -1.0])
            eta[k] = 0.0
            continue

        sd = sun_dir_enu(alt, az)
        sun_dirs[k] = sd

        # Sun must be in front of window  # 中文：太阳必须位于窗的正前方（与外法线同向一侧）
        if np.dot(sd, frame.n) <= 1e-9:
            eta[k] = 0.0
            continue

        sd_local = frame.world_to_local_vec(sd)
        shaded = scene.shaded_mask(sample_pts_local, sd_local)
        lit_frac = 1.0 - shaded.mean()
        eta[k] = float(lit_frac)

    return sun_dirs, eta

def evaluate_design(
    times: pd.DatetimeIndex,
    weather: pd.DataFrame,  # columns: Tout, DNI, DHI  # 中文：天气数据列：室外温度 Tout、直射 DNI、散射 DHI
    lat_deg: float,
    frames: dict[str, Frame],
    wins: dict[str, WindowSpec],
    Awin: dict[str, float],
    nx: int,
    ny: int,
    design: Design,
    # thermal params:  # 中文：热工参数
    UA_total: float,
    SHGC: float,
    k_diff: float,
    COP_cool: float,
    eta_heat: float,
    T_in: float,
    Qint_work: float,
    Qint_off: float,
    dt_hours: float,
    # comfort/daylight params:  # 中文：舒适性/采光参数
    glare_depth_thresh: float,
    glare_hours_max: float,
    work_start: int,
    work_end: int,
    lux_min: float,
    tau_diff: float,
    kappa: float,
    C_dl: float,
    k_diff_shade: float,
    floor_area: float,
    # objective weights:  # 中文：目标函数权重
    w_cool: float,
    w_heat: float,
    # baseline results for metrics:  # 中文：用于指标计算的基准结果
    baseline_J: float,
    baseline_Qcool_peak_kW: float,
) -> EvalOutputs:
    # build per-facade scenes & samples  # 中文：为每个立面构建遮阳场景与窗面采样点
    scenes = {}
    samples = {}
    for f in frames.keys():
        scene, pts_local = build_facade_scene_and_samples(f, frames[f], wins[f], nx, ny, design)
        scenes[f] = scene
        samples[f] = pts_local

    # compute sun dirs and eta for each facade, also facade irradiance  # 中文：计算太阳方向与 eta，并计算各立面辐照
    Tout = weather["Tout"].to_numpy(float)
    DNI = weather["DNI"].to_numpy(float)
    DHI = weather["DHI"].to_numpy(float)

    sun_dirs_world = None
    eta_facade = {}
    Idir_facade = {}
    Idiff_facade = {}

    for f, frame in frames.items():
        sun_dirs, eta = compute_eta_series(times, lat_deg, frame, scenes[f], samples[f])
        eta_facade[f] = eta
        if sun_dirs_world is None:
            sun_dirs_world = sun_dirs

        # irradiance on this facade  # 中文：该立面的辐照度（直射与散射）
        Idir = np.zeros_like(DNI)
        Idiff = np.zeros_like(DHI)
        # facade azimuth is encoded by frame.n; recover azimuth for irradiance calc:  # 中文：立面方位由 frame.n 隐含，先恢复方位角以便计算入射
        # az = atan2(n_x, n_y)  # 中文：方位角 az 的恢复方式（由法线分量反推）
        az = np.rad2deg(np.arctan2(frame.n[0], frame.n[1])) % 360.0

        for k in range(len(times)):
            Idir[k] = facade_dir_irradiance(DNI[k], sun_dirs_world[k], az)
            Idiff[k] = facade_diffuse_irradiance(DHI[k])

        Idir_facade[f] = Idir
        Idiff_facade[f] = Idiff

    # comfort constraints  # 中文：舒适性约束（眩光/采光）
    glare_res = compute_glare_hours(
        times, frames, wins, samples, scenes,
        sun_dirs_world, dt_hours,
        work_start, work_end,
        glare_depth_thresh
    )
    glare_ok = (glare_res.glare_hours <= glare_hours_max)

    dl_ok, E_in = daylight_ok(
        times, DHI,
        Awin_by_facade=Awin,
        tau_diff=tau_diff,
        k_diff=k_diff_shade,
        floor_area=floor_area,
        kappa=kappa,
        C_dl=C_dl,
        work_start=work_start,
        work_end=work_end,
        lux_min=lux_min
    )

    # energy  # 中文：能耗计算
    energy = compute_energy(
        times, Tout,
        Idir_facade, Idiff_facade, eta_facade,
        Awin, UA_total,
        SHGC, k_diff,
        COP_cool, eta_heat,
        T_in,
        work_start, work_end,
        Qint_work, Qint_off,
        dt_hours,
        w_cool, w_heat
    )

    feasible = glare_ok and dl_ok

    # metrics (vs baseline)  # 中文：与基准方案对比的指标
    AESR = (baseline_J - energy.J) / max(1e-9, baseline_J) * 100.0
    PLR = (baseline_Qcool_peak_kW - energy.Qcool_peak_kW) / max(1e-9, baseline_Qcool_peak_kW) * 100.0

    return EvalOutputs(
        feasible=feasible,
        glare_hours=glare_res.glare_hours,
        daylight_ok=dl_ok,
        energy=energy,
        AESR=float(AESR),
        PLR=float(PLR),
    )
