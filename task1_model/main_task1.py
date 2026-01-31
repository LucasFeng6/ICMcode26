# task1_model/main_task1.py
import numpy as np
import pandas as pd

from task1_model.config import (
    BuildingGeom, HVACParams, EnvelopeParams, InternalGains,
    DaylightingParams, ComfortConstraints, OptimizationWeights
)
from task1_model.frames import Frame
from task1_model.sampling import WindowSpec
from task1_model.evaluate import Design, evaluate_design
from task1_model.optimize import grid_search

def load_weather_csv(path: str) -> tuple[pd.DatetimeIndex, pd.DataFrame, float]:
    """
    CSV required columns:
      datetime (ISO), Tout (degC), DNI (W/m2), DHI (W/m2)
    Also set dt_hours from time step.
    """
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    times = pd.DatetimeIndex(df["datetime"])
    w = df[["Tout", "DNI", "DHI"]].copy()
    # infer dt hours
    if len(times) >= 2:
        dt_hours = (times[1] - times[0]).total_seconds() / 3600.0
    else:
        dt_hours = 1.0
    return times, w, float(dt_hours)

def compute_UA_total(bg: BuildingGeom, env: EnvelopeParams) -> tuple[float, dict[str, float]]:
    # facade areas
    A_S = bg.L * bg.H
    A_N = bg.L * bg.H
    A_E = bg.W * bg.H
    A_W = bg.W * bg.H
    Awin = {
        "S": bg.wwr_south * A_S,
        "N": bg.wwr_other * A_N,
        "E": bg.wwr_other * A_E,
        "W": bg.wwr_other * A_W,
    }
    Aop = {
        "S": A_S - Awin["S"],
        "N": A_N - Awin["N"],
        "E": A_E - Awin["E"],
        "W": A_W - Awin["W"],
    }
    UA = env.U_wall * sum(Aop.values()) + env.U_win * sum(Awin.values())
    return float(UA), Awin

def main():
    # -------------------------
    # User inputs (edit here)
    # -------------------------
    weather_path = "weather_tmy.csv"  # <-- your file
    lat_deg = 15.0                   # Sungrove approx low latitude; replace with actual if known

    # Representative window geometry (per facade) - EDIT these
    # You said: window height above floor known; width/height known; only one window per facade.
    # NOTE: sill_h is bottom of window above that floor. If you only know "window elevation above ground",
    # set floor_z=0 and sill_h accordingly for 1F; for 2F set floor_z=floor_height.
    floor_height = 4.0
    wins = {
        "S": WindowSpec(width=3.0, height=2.0, sill_h=1.0, floor_z=0.0),
        "N": WindowSpec(width=3.0, height=2.0, sill_h=1.0, floor_z=0.0),
        "E": WindowSpec(width=3.0, height=2.0, sill_h=1.0, floor_z=0.0),
        "W": WindowSpec(width=3.0, height=2.0, sill_h=1.0, floor_z=0.0),
    }

    # Window frames (origins can be (0,0,floor_z) since we use local-only scenes)
    # For ENU coords: x=E, y=N, z=Up. Origins only need z correct for glare height definition.
    frames = {
        "N": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, wins["N"].floor_z]), facade_azimuth_deg=0.0),
        "E": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, wins["E"].floor_z]), facade_azimuth_deg=90.0),
        "S": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, wins["S"].floor_z]), facade_azimuth_deg=180.0),
        "W": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, wins["W"].floor_z]), facade_azimuth_deg=270.0),
    }

    # Sampling resolution
    nx, ny = 12, 12

    # -------------------------
    # Load config
    # -------------------------
    bg = BuildingGeom()
    hvac = HVACParams()
    env = EnvelopeParams()
    gains = InternalGains()
    dl = DaylightingParams()
    cc = ComfortConstraints()
    w = OptimizationWeights()

    # -------------------------
    # Load weather
    # -------------------------
    times, weather, dt_hours = load_weather_csv(weather_path)

    # Building areas / UA
    UA_total, Awin = compute_UA_total(bg, env)
    floor_area = bg.L * bg.W * 2.0

    # Choose Tin (task1 simplified): fixed comfort temp (you can improve with seasonal schedule)
    T_in = 24.0  # consistent with cooling-dominant site; you can set per season later

    # -------------------------
    # Baseline evaluation (no shading)
    # -------------------------
    baseline = Design(D_oh=0.0, D_fin=0.0, beta_fin_deg=0.0)
    base_out = evaluate_design(
        times, weather, lat_deg,
        frames, wins, Awin,
        nx, ny, baseline,
        UA_total, env.SHGC, dl.k_diff_shade, hvac.COP_cool, hvac.eta_heat, T_in,
        gains.Q_internal_work, gains.Q_internal_off, dt_hours,
        cc.glare_depth_m, cc.glare_hours_max, cc.work_start_hour, cc.work_end_hour,
        cc.daylight_lux_min, env.tau_diff, dl.kappa, dl.C_dl, dl.k_diff_shade, floor_area,
        w.w_cool, w.w_heat,
        baseline_J=1.0,  # placeholder; overwrite below
        baseline_Qcool_peak_kW=1.0
    )
    baseline_J = base_out.energy.J
    baseline_peak = base_out.energy.Qcool_peak_kW

    # re-evaluate baseline with correct baseline refs for metrics
    base_out = evaluate_design(
        times, weather, lat_deg,
        frames, wins, Awin,
        nx, ny, baseline,
        UA_total, env.SHGC, dl.k_diff_shade, hvac.COP_cool, hvac.eta_heat, T_in,
        gains.Q_internal_work, gains.Q_internal_off, dt_hours,
        cc.glare_depth_m, cc.glare_hours_max, cc.work_start_hour, cc.work_end_hour,
        cc.daylight_lux_min, env.tau_diff, dl.kappa, dl.C_dl, dl.k_diff_shade, floor_area,
        w.w_cool, w.w_heat,
        baseline_J=baseline_J,
        baseline_Qcool_peak_kW=baseline_peak
    )

    print("=== Baseline ===")
    print("Feasible:", base_out.feasible, "| Glare hours:", base_out.glare_hours, "| Daylight OK:", base_out.daylight_ok)
    print("E_cool(kWh):", base_out.energy.E_cool_kWh, "E_heat(kWh):", base_out.energy.E_heat_kWh,
          "Peak(kW):", base_out.energy.Qcool_peak_kW, "J:", base_out.energy.J)

    # -------------------------
    # Optimization grid (task 1)
    # -------------------------
    D_oh_list = np.linspace(0.0, 2.0, 11)        # 0..2m
    D_fin_list = np.linspace(0.0, 1.0, 11)       # 0..1m
    beta_list = np.arange(0.0, 91.0, 15.0)       # 0..90 deg

    best = grid_search(
        times, weather, lat_deg,
        frames, wins, Awin,
        nx, ny,
        UA_total, env.SHGC, dl.k_diff_shade, hvac.COP_cool, hvac.eta_heat, T_in,
        gains.Q_internal_work, gains.Q_internal_off,
        dt_hours,
        cc.glare_depth_m, cc.glare_hours_max, cc.work_start_hour, cc.work_end_hour,
        cc.daylight_lux_min,
        env.tau_diff, dl.kappa, dl.C_dl, dl.k_diff_shade, floor_area,
        w.w_cool, w.w_heat,
        baseline_J, baseline_peak,
        D_oh_list, D_fin_list, beta_list
    )

    print("\n=== Best feasible retrofit ===")
    print("Design:", best.design)
    print("Glare hours:", best.outputs.glare_hours, "(<= 50)")
    print("Daylight OK:", best.outputs.daylight_ok, "(>= 300 lux in work hours)")
    print("E_cool(kWh):", best.outputs.energy.E_cool_kWh,
          "E_heat(kWh):", best.outputs.energy.E_heat_kWh,
          "Peak(kW):", best.outputs.energy.Qcool_peak_kW,
          "J:", best.outputs.energy.J)
    print("AESR(%):", best.outputs.AESR, "PLR(%):", best.outputs.PLR)

if __name__ == "__main__":
    main()
