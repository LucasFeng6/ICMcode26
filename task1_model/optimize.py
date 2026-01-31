# task1_model/optimize.py  # 中文：简单网格搜索优化
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .evaluate import Design, build_facade_scene_and_samples, EvalOutputs
from .comfort import glare_depth_from_samples, daylight_ok
from .thermal import compute_energy
from .solar_geometry import (
    solar_alt_az_deg,
    sun_dir_enu,
    facade_dir_irradiance,
    facade_diffuse_irradiance,
    within_work_hours,
)

def _compute_sun_dirs_world(times: pd.DatetimeIndex, lat_deg: float) -> tuple[np.ndarray, np.ndarray]:
    T = len(times)
    sun_dirs = np.zeros((T, 3), float)
    sun_up = np.zeros((T,), dtype=bool)
    for k, ts in enumerate(times):
        n = int(ts.dayofyear)
        local_hour = ts.hour + ts.minute / 60.0
        alt, az = solar_alt_az_deg(lat_deg, n, local_hour)
        if alt <= 0.0:
            sun_dirs[k] = np.array([0.0, 0.0, -1.0])
            sun_up[k] = False
            continue
        sun_dirs[k] = sun_dir_enu(alt, az)
        sun_up[k] = True
    return sun_dirs, sun_up

def _compute_facade_irradiance(
    frames,
    DNI: np.ndarray,
    DHI: np.ndarray,
    sun_dirs_world: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    Idir_facade: dict[str, np.ndarray] = {}
    Idiff_facade: dict[str, np.ndarray] = {}
    for f, frame in frames.items():
        az = np.rad2deg(np.arctan2(frame.n[0], frame.n[1])) % 360.0
        Idir = np.zeros_like(DNI)
        Idiff = np.zeros_like(DHI)
        for k in range(len(DNI)):
            Idir[k] = facade_dir_irradiance(DNI[k], sun_dirs_world[k], az)
            Idiff[k] = facade_diffuse_irradiance(DHI[k])
        Idir_facade[f] = Idir
        Idiff_facade[f] = Idiff
    return Idir_facade, Idiff_facade

def _compute_eta_and_glare_depth(
    times: pd.DatetimeIndex,
    frame,
    win,
    scene,
    sample_pts_local: np.ndarray,
    sun_dirs_world: np.ndarray,
    sun_up: np.ndarray,
    work_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(times)
    eta = np.zeros((T,), float)
    glare_depth = np.zeros((T,), float)

    for k in range(T):
        if not sun_up[k]:
            continue
        sun_dir = sun_dirs_world[k]
        if np.dot(sun_dir, frame.n) <= 1e-9:
            continue

        sun_local = frame.world_to_local_vec(sun_dir)
        shaded = scene.shaded_mask(sample_pts_local, sun_local)
        eta[k] = float(1.0 - shaded.mean())

        if work_mask[k]:
            glare_depth[k] = glare_depth_from_samples(frame, win, sample_pts_local, shaded, sun_dir)

    return eta, glare_depth

@dataclass(frozen=True)
class BestResult:
    design: Design
    outputs: EvalOutputs

def grid_search(
    times: pd.DatetimeIndex,
    weather: pd.DataFrame,
    lat_deg: float,
    frames,
    wins,
    Awin,
    nx: int,
    ny: int,
    UA_total: float,
    SHGC: float,
    k_diff: float,
    COP_cool: float,
    eta_heat: float,
    T_in: float,
    Qint_work: float,
    Qint_off: float,
    dt_hours: float,
    glare_depth_thresh: float,
    glare_hours_max: float,
    work_start: int,
    work_end: int,
    lux_min: float,
    daylight_ok_hours_min: float,
    tau_diff: float,
    kappa: float,
    C_dl: float,
    C_dir: float,
    k_diff_shade: float,
    floor_area: float,
    w_cool: float,
    w_heat: float,
    baseline_J: float,
    baseline_Qcool_peak_kW: float,
    D_oh_list,
    fin_w_list,
    beta_list,
    progress: bool = True,
    progress_every: int = 10,
) -> BestResult:
    best = None
    best_any = None

    total = len(D_oh_list) * len(fin_w_list) * len(beta_list)
    start = time.perf_counter()
    if progress:
        if progress_every <= 0:
            progress_every = 1
        print(f"[Grid] start: {total} designs")
        if total == 0:
            print("[Grid] no designs to evaluate")

    i = 0

    # Precompute items independent of design
    Tout = weather["Tout"].to_numpy(float)
    DNI = weather["DNI"].to_numpy(float)
    DHI = weather["DHI"].to_numpy(float)

    sun_dirs_world, sun_up = _compute_sun_dirs_world(times, lat_deg)
    work_mask = np.array(
        [within_work_hours(ts, work_start, work_end) for ts in times],
        dtype=bool,
    )

    Idir_facade, Idiff_facade = _compute_facade_irradiance(frames, DNI, DHI, sun_dirs_world)

    # Cache per-facade results for separable design variables
    south_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    if progress:
        print(f"[Cache] south facade: {len(D_oh_list)} variants")
    south_i = 0
    for D_oh in D_oh_list:
        d = Design(D_oh=float(D_oh), fin_w=0.0, beta_fin_deg=0.0)
        scene, pts_local = build_facade_scene_and_samples("S", frames["S"], wins["S"], nx, ny, d)
        eta, glare_depth = _compute_eta_and_glare_depth(
            times, frames["S"], wins["S"], scene, pts_local, sun_dirs_world, sun_up, work_mask
        )
        south_cache[float(D_oh)] = (eta, glare_depth)
        south_i += 1
        if progress and (south_i % progress_every == 0 or south_i == len(D_oh_list)):
            elapsed = time.perf_counter() - start
            pct = (south_i / max(1, len(D_oh_list)) * 100.0)
            print(f"[Cache] south {south_i}/{len(D_oh_list)} ({pct:.0f}%) elapsed {elapsed:.1f}s")

    fin_cache: dict[tuple[float, float], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    total_fin = len(fin_w_list) * len(beta_list)
    if progress:
        print(f"[Cache] fins (N/E/W): {total_fin} variants")
    fin_i = 0
    for fin_w in fin_w_list:
        for beta in beta_list:
            key = (float(fin_w), float(beta))
            eta_fac: dict[str, np.ndarray] = {}
            glare_fac: dict[str, np.ndarray] = {}
            d = Design(D_oh=0.0, fin_w=float(fin_w), beta_fin_deg=float(beta))
            for f in ("N", "E", "W"):
                scene, pts_local = build_facade_scene_and_samples(f, frames[f], wins[f], nx, ny, d)
                eta, glare_depth = _compute_eta_and_glare_depth(
                    times, frames[f], wins[f], scene, pts_local, sun_dirs_world, sun_up, work_mask
                )
                eta_fac[f] = eta
                glare_fac[f] = glare_depth
            fin_cache[key] = (eta_fac, glare_fac)
            fin_i += 1
            if progress and (fin_i % progress_every == 0 or fin_i == total_fin):
                elapsed = time.perf_counter() - start
                pct = (fin_i / max(1, total_fin) * 100.0)
                print(f"[Cache] fins {fin_i}/{total_fin} ({pct:.0f}%) elapsed {elapsed:.1f}s")

    for D_oh in D_oh_list:
        for fin_w in fin_w_list:
            for beta in beta_list:
                i += 1
                if progress and (i % progress_every == 0 or i == total):
                    elapsed = time.perf_counter() - start
                    pct = (i / total * 100.0) if total else 100.0
                    print(f"[Grid] {i}/{total} ({pct:.0f}%) elapsed {elapsed:.1f}s")

                eta_S, glare_S = south_cache[float(D_oh)]
                eta_fac, glare_fac = fin_cache[(float(fin_w), float(beta))]
                eta_facade = {"S": eta_S, **eta_fac}

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
                    lux_min=lux_min,
                    dt_hours=dt_hours,
                    ok_hours_min=daylight_ok_hours_min,
                    Idir_facade=Idir_facade,
                    eta_facade=eta_facade,
                    C_dir=C_dir,
                )
                daylight_ok_hours = float(((work_mask) & (E_in >= lux_min)).sum() * dt_hours)
                glare_max = np.maximum.reduce(
                    [glare_S, glare_fac["N"], glare_fac["E"], glare_fac["W"]]
                )
                glare_hours = float((glare_max > glare_depth_thresh).sum() * dt_hours)
                glare_ok_hours = float(((work_mask) & (glare_max <= glare_depth_thresh)).sum() * dt_hours)
                glare_ok = glare_hours <= glare_hours_max

                feasible = bool(glare_ok and dl_ok)

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
                    w_cool, w_heat,
                )

                AESR = (baseline_J - energy.J) / max(1e-9, baseline_J) * 100.0
                PLR = (baseline_Qcool_peak_kW - energy.Qcool_peak_kW) / max(1e-9, baseline_Qcool_peak_kW) * 100.0

                d = Design(D_oh=float(D_oh), fin_w=float(fin_w), beta_fin_deg=float(beta))
                out = EvalOutputs(
                    feasible=feasible,
                    glare_hours=glare_hours,
                    glare_ok_hours=glare_ok_hours,
                    daylight_ok=dl_ok,
                    daylight_ok_hours=daylight_ok_hours,
                    energy=energy,
                    AESR=float(AESR),
                    PLR=float(PLR),
                )

                if feasible and ((best is None) or (out.energy.J < best.outputs.energy.J)):
                    best = BestResult(d, out)
                if (best_any is None) or (out.energy.J < best_any.outputs.energy.J):
                    best_any = BestResult(d, out)

    if best is not None:
        return best
    if best_any is not None:
        return best_any
    raise RuntimeError("No feasible design found. Relax bounds or check daylight/glare parameters.")
