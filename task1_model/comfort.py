# task1_model/comfort.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .frames import Frame
from .ray_scene import RayScene
from .sampling import WindowSpec
from .solar_geometry import within_work_hours

@dataclass(frozen=True)
class GlareResult:
    glare_hours: float
    glare_mask: np.ndarray  # per timestep bool
    max_depth_per_t: np.ndarray  # per timestep (m)

def glare_depth_from_samples(
    frame: Frame,
    win: WindowSpec,
    sample_pts_local: np.ndarray,
    shaded_mask: np.ndarray,
    sun_dir_world: np.ndarray,
) -> float:
    """
    Glare depth at one timestep for one facade.
    - shaded_mask True means blocked; False means lit
    - pick lit point with max z (local v), compute depth based on entering direction.
    """
    # Sun must be in front of window: dot(sun_dir, n_out) > 0
    if np.dot(sun_dir_world, frame.n) <= 1e-9:
        return 0.0

    lit_idx = np.where(~shaded_mask)[0]
    if lit_idx.size == 0:
        return 0.0

    # Choose lit point with maximum local v
    lit_pts = sample_pts_local[lit_idx]
    j = np.argmax(lit_pts[:, 1])  # local v
    p_star = lit_pts[j]

    # Height above the floor of that level:
    h_star = (win.floor_z + p_star[1]) - win.floor_z  # simplifies to p_star[1]
    h_star = float(h_star)

    # Entering light direction is from outside to inside: -sun_dir_local
    sun_dir_local = frame.world_to_local_vec(sun_dir_world)
    d_enter = -sun_dir_local  # into room: n component should be negative (inward)

    # Vertical component is local v
    dv = d_enter[1]
    dn = d_enter[2]  # along local n (outward). Inward is negative.
    if dv >= -1e-9:
        # light not going downward to floor
        return 0.0

    depth = h_star * (abs(dn) / (abs(dv) + 1e-12))
    return float(depth)

def compute_glare_hours(
    times: pd.DatetimeIndex,
    frames: dict[str, Frame],
    wins: dict[str, WindowSpec],
    samples_local: dict[str, np.ndarray],
    scenes: dict[str, RayScene],
    sun_dirs_world: np.ndarray,   # (T,3)
    dt_hours: float,
    work_start: int,
    work_end: int,
    glare_depth_thresh: float,
) -> GlareResult:
    """
    A timestep is "glare" if max facade glare depth > thresh during work hours.
    """
    T = len(times)
    glare_mask = np.zeros((T,), dtype=bool)
    max_depth = np.zeros((T,), dtype=float)

    for k, ts in enumerate(times):
        if not within_work_hours(ts, work_start, work_end):
            continue

        sun_dir = sun_dirs_world[k]
        dmax = 0.0

        for f in frames.keys():
            frame = frames[f]
            win = wins[f]
            pts = samples_local[f]

            # if sun behind facade, skip quickly
            if np.dot(sun_dir, frame.n) <= 1e-9:
                continue

            sun_local = frame.world_to_local_vec(sun_dir)
            shaded = scenes[f].shaded_mask(pts, sun_local)

            d = glare_depth_from_samples(frame, win, pts, shaded, sun_dir)
            if d > dmax:
                dmax = d

        max_depth[k] = dmax
        if dmax > glare_depth_thresh:
            glare_mask[k] = True

    glare_hours = float(glare_mask.sum() * dt_hours)
    return GlareResult(glare_hours=glare_hours, glare_mask=glare_mask, max_depth_per_t=max_depth)

def daylight_ok(
    times: pd.DatetimeIndex,
    DHI: np.ndarray,
    Awin_by_facade: dict[str, float],
    tau_diff: float,
    k_diff: float,
    floor_area: float,
    kappa: float,
    C_dl: float,
    work_start: int,
    work_end: int,
    lux_min: float
) -> tuple[bool, np.ndarray]:
    """
    Diffuse-only daylight model:
      E_in_avg(t) = C_dl * kappa * DHI(t) * sum(tau*A*k_diff) / A_floor
    Returns (all_ok, E_in_series)
    """
    trans_area = sum(A * tau_diff * k_diff for A in Awin_by_facade.values())
    E_in = C_dl * kappa * np.maximum(0.0, DHI) * (trans_area / max(1e-9, floor_area))

    ok = np.ones((len(times),), dtype=bool)
    for k, ts in enumerate(times):
        if within_work_hours(ts, work_start, work_end):
            ok[k] = (E_in[k] >= lux_min)
    return bool(ok.all()), E_in
