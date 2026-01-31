"""
Export baseline (no passive shading devices) glare + illuminance time series to CSV.

Run from project root:
  python export_baseline_comfort.py --weather "E:\\path\\weather.csv" --out "baseline_comfort.csv"
or:
  python -m export_baseline_comfort --weather "E:\\path\\weather.csv" --out "baseline_comfort.csv"

Input weather CSV must include columns:
  datetime, Tout, DNI, DHI

Notes:
  - Glare uses this project's "glare depth" proxy (meters), not DGP/UGR.
  - Illuminance uses a diffuse-only daylight model mapped from DHI (lux).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd

from task1_model.comfort import daylight_ok, glare_depth_from_samples
from task1_model.config import BuildingGeom, ComfortConstraints, DaylightingParams, EnvelopeParams
from task1_model.frames import Frame
from task1_model.main_task1 import compute_UA_total, load_weather_csv
from task1_model.sampling import WindowSpec, rect_grid_samples_local
from task1_model.solar_geometry import solar_alt_az_deg, sun_dir_enu, within_work_hours


def _sun_dirs_world(times: pd.DatetimeIndex, lat_deg: float) -> np.ndarray:
    sun_dirs = np.zeros((len(times), 3), float)
    for k, ts in enumerate(times):
        n = int(ts.dayofyear)
        local_hour = ts.hour + ts.minute / 60.0
        alt, az = solar_alt_az_deg(lat_deg, n, local_hour)
        if alt <= 0.0:
            sun_dirs[k] = np.array([0.0, 0.0, -1.0])
        else:
            sun_dirs[k] = sun_dir_enu(alt, az)
    return sun_dirs


def compute_baseline_series(
    times: pd.DatetimeIndex,
    weather: pd.DataFrame,
    lat_deg: float,
    dt_hours: float,
    nx: int,
    ny: int,
    win: WindowSpec,
    glare_depth_thresh_m: float,
    work_start_hour: int,
    work_end_hour: int,
    lux_min: float,
    tau_diff: float,
    kappa: float,
    C_dl: float,
    k_diff: float,
    Awin_by_facade: dict[str, float],
    floor_area: float,
) -> pd.DataFrame:
    facades = ("N", "E", "S", "W")
    frames = {
        "N": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, win.floor_z]), facade_azimuth_deg=0.0),
        "E": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, win.floor_z]), facade_azimuth_deg=90.0),
        "S": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, win.floor_z]), facade_azimuth_deg=180.0),
        "W": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, win.floor_z]), facade_azimuth_deg=270.0),
    }

    sample_pts_local = rect_grid_samples_local(win, nx=nx, ny=ny)
    shaded_none = np.zeros((sample_pts_local.shape[0],), dtype=bool)

    sun_dirs_world = _sun_dirs_world(times, lat_deg)

    # Per-facade glare depth series (baseline: no shading devices => all points lit)
    glare_depth_by_facade: dict[str, np.ndarray] = {}
    for f in facades:
        frame = frames[f]
        depth = np.zeros((len(times),), float)
        for k in range(len(times)):
            depth[k] = glare_depth_from_samples(frame, win, sample_pts_local, shaded_none, sun_dirs_world[k])
        glare_depth_by_facade[f] = depth

    glare_depth_max = np.maximum.reduce([glare_depth_by_facade[f] for f in facades])
    work_mask = np.array(
        [within_work_hours(ts, work_start_hour, work_end_hour) for ts in times],
        dtype=bool,
    )
    glare_flag = (glare_depth_max > glare_depth_thresh_m) & work_mask

    # Daylight / illuminance series (diffuse-only)
    DHI = weather["DHI"].to_numpy(float)
    _, E_in = daylight_ok(
        times,
        DHI,
        Awin_by_facade=Awin_by_facade,
        tau_diff=tau_diff,
        k_diff=k_diff,
        floor_area=floor_area,
        kappa=kappa,
        C_dl=C_dl,
        work_start=work_start_hour,
        work_end=work_end_hour,
        lux_min=lux_min,
        dt_hours=dt_hours,
        ok_hours_min=None,
    )
    daylight_ok_t = (~work_mask) | (E_in >= lux_min)

    out = pd.DataFrame(
        {
            "datetime": times.astype("datetime64[ns]"),
            "is_work_hour": work_mask.astype(int),
            "glare_depth_N_m": glare_depth_by_facade["N"],
            "glare_depth_E_m": glare_depth_by_facade["E"],
            "glare_depth_S_m": glare_depth_by_facade["S"],
            "glare_depth_W_m": glare_depth_by_facade["W"],
            "glare_depth_max_m": glare_depth_max,
            "glare_flag": glare_flag.astype(int),
            "E_in_lux": E_in,
            "daylight_ok_t": daylight_ok_t.astype(int),
        }
    )

    # CSV readability: keep 1 decimal for glare depth and illuminance.
    out = out.round(
        {
            "glare_depth_N_m": 1,
            "glare_depth_E_m": 1,
            "glare_depth_S_m": 1,
            "glare_depth_W_m": 1,
            "glare_depth_max_m": 1,
            "E_in_lux": 1,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline glare + illuminance per timestep to CSV.")
    parser.add_argument("--weather", type=str, required=True, help="Input weather CSV path.")
    parser.add_argument("--out", type=str, default="baseline_comfort.csv", help="Output CSV path.")

    # Keep defaults aligned with task1_model/main_task1.py example
    parser.add_argument("--lat", type=float, default=15.0, help="Latitude (deg).")
    parser.add_argument("--nx", type=int, default=12, help="Window sample count in u direction.")
    parser.add_argument("--ny", type=int, default=12, help="Window sample count in v direction.")
    parser.add_argument("--win-width", type=float, default=3.0, help="Representative window width (m).")
    parser.add_argument("--win-height", type=float, default=2.0, help="Representative window height (m).")
    parser.add_argument("--sill-h", type=float, default=1.0, help="Window sill height above floor (m).")
    parser.add_argument("--floor-z", type=float, default=0.0, help="World z of the window's floor level (m).")

    parser.add_argument("--work-start", type=int, default=8, help="Work start hour (inclusive).")
    parser.add_argument("--work-end", type=int, default=17, help="Work end hour (inclusive).")
    parser.add_argument("--glare-depth-thresh", type=float, default=1.5, help="Glare depth threshold (m).")
    parser.add_argument("--lux-min", type=float, default=300.0, help="Minimum illuminance during work hours (lux).")

    parser.add_argument("--tau-diff", type=float, default=None, help="Diffuse transmittance (overrides config).")
    parser.add_argument("--kappa", type=float, default=None, help="kappa (lux per W/m2) (overrides config).")
    parser.add_argument("--C-dl", type=float, default=None, help="C_dl factor (overrides config).")
    parser.add_argument("--k-diff", type=float, default=1.0, help="Diffuse shading factor (baseline should be 1.0).")

    args = parser.parse_args()

    times, weather, dt_hours = load_weather_csv(args.weather)

    bg = BuildingGeom()
    env = EnvelopeParams()
    dl = DaylightingParams()
    cc = ComfortConstraints()

    _UA_total, Awin = compute_UA_total(bg, env)
    floor_area = bg.L * bg.W * 2.0

    tau_diff = env.tau_diff if args.tau_diff is None else float(args.tau_diff)
    kappa = dl.kappa if args.kappa is None else float(args.kappa)
    C_dl = dl.C_dl if args.C_dl is None else float(args.C_dl)

    win = WindowSpec(
        width=float(args.win_width),
        height=float(args.win_height),
        sill_h=float(args.sill_h),
        floor_z=float(args.floor_z),
    )

    df = compute_baseline_series(
        times=times,
        weather=weather,
        lat_deg=float(args.lat),
        dt_hours=float(dt_hours),
        nx=int(args.nx),
        ny=int(args.ny),
        win=win,
        glare_depth_thresh_m=float(args.glare_depth_thresh),
        work_start_hour=int(args.work_start),
        work_end_hour=int(args.work_end),
        lux_min=float(args.lux_min),
        tau_diff=tau_diff,
        kappa=kappa,
        C_dl=C_dl,
        k_diff=float(args.k_diff),
        Awin_by_facade=Awin,
        floor_area=float(floor_area),
    )

    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    meta = {
        "weather": args.weather,
        "out": args.out,
        "lat_deg": float(args.lat),
        "dt_hours": float(dt_hours),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "win": asdict(win),
        "work_start": int(args.work_start),
        "work_end": int(args.work_end),
        "glare_depth_thresh_m": float(args.glare_depth_thresh),
        "lux_min": float(args.lux_min),
        "tau_diff": float(tau_diff),
        "kappa": float(kappa),
        "C_dl": float(C_dl),
        "k_diff": float(args.k_diff),
        "daylight_ok_hours_min_config": float(cc.daylight_ok_hours_min),
    }
    print(f"[OK] wrote: {args.out}")
    print("[Meta]", meta)


if __name__ == "__main__":
    main()

