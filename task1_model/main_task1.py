# 主程序入口（读取天气 -> 评估 -> 网格搜索优化）
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

    中文：
    CSV 必需列：
      datetime（ISO 时间），Tout（℃），DNI（W/m²），DHI（W/m²）。
    同时根据时间步长推断 dt_hours。
    """
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    times = pd.DatetimeIndex(df["datetime"])
    w = df[["Tout", "DNI", "DHI"]].copy()
    # infer dt hours  # 中文：推断时间步长（小时）
    if len(times) >= 2:
        dt_hours = (times[1] - times[0]).total_seconds() / 3600.0
    else:
        dt_hours = 1.0
    return times, w, float(dt_hours)

def compute_UA_total(bg: BuildingGeom, env: EnvelopeParams) -> tuple[float, dict[str, float]]:
    # 各立面面积
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
    # User inputs (edit here)  # 中文：用户输入（在此处修改）
    # -------------------------
    weather_path = "weather_tmy.csv"  # <-- your file  天气文件路径
    lat_deg = 15.0                   # Sungrove 低纬度近似值；若已知替换为真实纬度

    # 代表性窗户几何（按立面）——请在此处修改
    # 根据题意：已知窗离地高度、窗宽高；每个立面仅一个窗
    # 注意：sill_h 为窗底距该层楼面的高度；若只知道“窗离地高度”
    # 可将 1F 设 floor_z=0 并相应设置 sill_h；2F 则设 floor_z=floor_height
    floor_height = 4.0
    wins = {
        "S": WindowSpec(width=3.0, height=2.0, sill_h=1.0, floor_z=0.0),
        "N": WindowSpec(width=3.0, height=2.0, sill_h=1.0, floor_z=0.0),
        "E": WindowSpec(width=3.0, height=2.0, sill_h=1.0, floor_z=0.0),
        "W": WindowSpec(width=3.0, height=2.0, sill_h=1.0, floor_z=0.0),
    }

    # 窗户坐标框架（由于只用局部场景，原点可取 (0,0,floor_z)）
    # ENU 坐标：x=东、y=北、z=上；原点只需保证 z（楼层高度）正确即可用于眩光高度定义
    frames = {
        "N": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, wins["N"].floor_z]), facade_azimuth_deg=0.0),
        "E": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, wins["E"].floor_z]), facade_azimuth_deg=90.0),
        "S": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, wins["S"].floor_z]), facade_azimuth_deg=180.0),
        "W": Frame.from_facade_azimuth(origin=np.array([0.0, 0.0, wins["W"].floor_z]), facade_azimuth_deg=270.0),
    }

    # Sampling resolution  采样分辨率（窗面网格）
    nx, ny = 12, 12

    # -------------------------
    # 加载配置参数
    # -------------------------
    bg = BuildingGeom()
    hvac = HVACParams()
    env = EnvelopeParams()
    gains = InternalGains()
    dl = DaylightingParams()
    cc = ComfortConstraints()
    w = OptimizationWeights()

    # -------------------------
    # 读取天气数据
    # -------------------------
    times, weather, dt_hours = load_weather_csv(weather_path)

    # 建筑面积与 UA（总传热系数×面积）
    UA_total, Awin = compute_UA_total(bg, env)
    floor_area = bg.L * bg.W * 2.0

    # 选择室内温度 Tin（Task 1 简化）：固定舒适温度（可扩展为季节时间表）
    T_in = 24.0  # 与制冷主导地区一致；可后续按季节调整

    # -------------------------
    # Baseline evaluation (no shading)  # 中文：基准方案评估（无遮阳）
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
        baseline_J=1.0,  # 占位符；稍后用真实基准值覆盖
        baseline_Qcool_peak_kW=1.0
    )
    baseline_J = base_out.energy.J
    baseline_peak = base_out.energy.Qcool_peak_kW

    # re-evaluate baseline with correct baseline refs for metrics  # 中文：用正确的基准参考值重新评估（用于指标计算）
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
    # Optimization grid (task 1)  # 中文：优化网格（Task 1）
    # -------------------------
    D_oh_list = np.linspace(0.0, 2.0, 11)        # 0..2m  # 中文：挑檐深度范围 0..2 米
    D_fin_list = np.linspace(0.0, 1.0, 11)       # 0..1m  # 中文：侧翼深度范围 0..1 米
    beta_list = np.arange(0.0, 91.0, 15.0)       # 0..90 deg  # 中文：侧翼角度范围 0..90 度

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
