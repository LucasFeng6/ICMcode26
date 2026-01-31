# task1_model/optimize.py  # 中文：简单网格搜索优化
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .evaluate import Design, evaluate_design, EvalOutputs

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
    tau_diff: float,
    kappa: float,
    C_dl: float,
    k_diff_shade: float,
    floor_area: float,
    w_cool: float,
    w_heat: float,
    baseline_J: float,
    baseline_Qcool_peak_kW: float,
    D_oh_list,
    D_fin_list,
    beta_list,
    progress: bool = True,
    progress_every: int = 10,
) -> BestResult:
    best = None

    total = len(D_oh_list) * len(D_fin_list) * len(beta_list)
    start = time.perf_counter()
    if progress:
        if progress_every <= 0:
            progress_every = 1
        print(f"[Grid] start: {total} designs")
        if total == 0:
            print("[Grid] no designs to evaluate")

    i = 0

    for D_oh in D_oh_list:
        for D_fin in D_fin_list:
            for beta in beta_list:
                d = Design(D_oh=float(D_oh), D_fin=float(D_fin), beta_fin_deg=float(beta))
                out = evaluate_design(
                    times, weather, lat_deg,
                    frames, wins, Awin,
                    nx, ny, d,
                    UA_total, SHGC, k_diff, COP_cool, eta_heat, T_in,
                    Qint_work, Qint_off, dt_hours,
                    glare_depth_thresh, glare_hours_max, work_start, work_end,
                    lux_min, tau_diff, kappa, C_dl, k_diff_shade, floor_area,
                    w_cool, w_heat,
                    baseline_J, baseline_Qcool_peak_kW,
                )

                i += 1
                if progress and (i % progress_every == 0 or i == total):
                    elapsed = time.perf_counter() - start
                    pct = (i / total * 100.0) if total else 100.0
                    print(f"[Grid] {i}/{total} ({pct:.0f}%) elapsed {elapsed:.1f}s")

                if not out.feasible:
                    continue

                if (best is None) or (out.energy.J < best.outputs.energy.J):
                    best = BestResult(d, out)

    if best is None:
        raise RuntimeError("No feasible design found. Relax bounds or check daylight/glare parameters.")
    return best
