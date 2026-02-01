import os
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calplot

from solution_fast_v2 import (
    Config,
    load_and_preprocess,
    calc_shading_factors_fast,
    calc_diffuse_factor,
)


# =========================
# 用户可配置参数
# =========================
# 设计参数: (D_oh, D_fin, beta, SHGC_S, SHGC_Other)
DESIGN_PARAMS_BEFORE = (0.0, 0.0, 0.0, 0.6, 0.6)
DESIGN_PARAMS_AFTER = (1.6, 0.8, 60.0, 0.4, 0.3)

# 立面构件配置: (has_overhang, has_fin)
# 默认与 solution_fast_v2 的 facade_info 逻辑一致
FACADE_FLAGS: Dict[str, Tuple[bool, bool]] = {
    "S": (True, True),
    "N": (True, False),
    "E": (False, True),
    "W": (False, True),
}

# 气象数据路径
WEATHER_PATH = r"E:\数模\26美赛\数据\weather_Miami Intl Ap.csv"
LAT_DEG = 25


def get_window_areas():
    A_facade_S, A_facade_N = Config.L * Config.H, Config.L * Config.H
    A_facade_EW = Config.W * Config.H
    A_win_S = A_facade_S * Config.WWR_S
    A_win_N = A_facade_N * Config.WWR_N
    A_win_E = A_facade_EW * Config.WWR_EW
    A_win_W = A_facade_EW * Config.WWR_EW
    return A_win_S, A_win_N, A_win_E, A_win_W


def build_facade_info(flags: Dict[str, Tuple[bool, bool]]):
    A_win_S, A_win_N, A_win_E, A_win_W = get_window_areas()
    s_oh, s_fin = flags["S"]
    n_oh, n_fin = flags["N"]
    e_oh, e_fin = flags["E"]
    w_oh, w_fin = flags["W"]
    return [
        (A_win_S, 180, s_oh, s_fin, True),   # South
        (A_win_N, 0,   n_oh, n_fin, False),  # North
        (A_win_E, 90,  e_oh, e_fin, False),  # East
        (A_win_W, 270, w_oh, w_fin, False),  # West
    ]


def compute_hourly_masks(design_params, weather_data, sun_vecs, work_mask, facade_info):
    D_oh, D_fin, beta, SHGC_s, SHGC_other = design_params

    Q_solar_total = np.zeros(len(weather_data))
    glare_mask_year = np.zeros(len(weather_data), dtype=bool)
    E_in_diff = np.zeros(len(weather_data))

    for (A_win, az, has_oh, use_fin, is_south) in facade_info:
        SHGC_val = SHGC_s if is_south else SHGC_other
        d_oh_curr = D_oh if has_oh else 0.0
        d_fin_curr = D_fin if use_fin else 0.0
        beta_curr = beta if use_fin else 0.0
        k_diff_dyn = calc_diffuse_factor(d_oh_curr, d_fin_curr, beta_curr, Config.win_width, Config.win_height)

        eta, depth = calc_shading_factors_fast(
            sun_vecs, az, d_oh_curr, d_fin_curr, beta_curr,
            Config.win_width, Config.win_height
        )

        f_rad = np.deg2rad(az)
        cos_inc = sun_vecs[:, 0] * np.sin(f_rad) + sun_vecs[:, 1] * np.cos(f_rad)
        I_dir_outdoor = weather_data["DNI"].values * np.maximum(0, cos_inc)
        I_diff_outdoor = weather_data["DHI"].values * 0.5

        I_trans_dir = I_dir_outdoor * eta * SHGC_val

        # 眩光判定（与 solution_fast_v2 一致）
        is_glare = (
            work_mask
            & (depth > Config.glare_depth_limit)
            & (I_trans_dir > 50.0)
            & (eta > 0.05)
        )
        glare_mask_year |= is_glare

        # 采光（与 solution_fast_v2 一致）
        E_in_diff += (weather_data["DHI"].values * k_diff_dyn * A_win * SHGC_val)

    floor_area = Config.L * Config.W * 2
    E_in_diff = (E_in_diff / floor_area) * Config.kappa_lux

    daylight_ok = (E_in_diff >= Config.lux_min) & work_mask
    glare_occurs = work_mask & glare_mask_year

    return glare_occurs, daylight_ok


def aggregate_daily_hours(df, mask):
    tmp = df.copy()
    tmp["date"] = tmp["dt"].dt.date
    tmp["ok"] = mask.astype(int)
    daily = tmp.groupby("date")["ok"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily


def plot_calendar_heatmaps(glare_daily, daylight_daily, year):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    vmin = 0
    vmax = float(np.nanmax([glare_daily.values.max(), daylight_daily.values.max()]))
    if not np.isfinite(vmax):
        vmax = 1.0

    calplot.yearplot(
        glare_daily,
        year=year,
        ax=axes[0],
        cmap="YlGn",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Glare Hours (per day)")

    calplot.yearplot(
        daylight_daily,
        year=year,
        ax=axes[1],
        cmap="YlGn",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Daylight Constraint Hours (per day)")

    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap="YlGn")
    mappable.set_array([])
    fig.subplots_adjust(left=0.06, right=0.82, top=0.95, bottom=0.08, hspace=0.08)
    cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(mappable, cax=cax, label="Hours")
    return fig, axes


def plot_glare_before_after(glare_before, glare_after, year):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    vmin = 0
    vmax = float(np.nanmax([glare_before.values.max(), glare_after.values.max()]))
    if not np.isfinite(vmax):
        vmax = 1.0

    calplot.yearplot(
        glare_before,
        year=year,
        ax=axes[0],
        cmap="YlGn",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Glare Hours (Before)")

    calplot.yearplot(
        glare_after,
        year=year,
        ax=axes[1],
        cmap="YlGn",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Glare Hours (After)")

    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap="YlGn")
    mappable.set_array([])
    fig.subplots_adjust(left=0.06, right=0.82, top=0.95, bottom=0.08, hspace=0.08)
    cax = fig.add_axes([0.85, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
    fig.colorbar(mappable, cax=cax, label="Hours")
    return fig, axes


def run_calendar_plot(mode: int):
    df, sun_vecs, work_mask = load_and_preprocess(WEATHER_PATH, lat_deg=LAT_DEG)
    facade_info = build_facade_info(FACADE_FLAGS)
    glare_occurs_before, daylight_ok_before = compute_hourly_masks(
        DESIGN_PARAMS_BEFORE, df, sun_vecs, work_mask, facade_info
    )
    glare_occurs_after, daylight_ok_after = compute_hourly_masks(
        DESIGN_PARAMS_AFTER, df, sun_vecs, work_mask, facade_info
    )

    year = int(df["dt"].dt.year.mode()[0])
    glare_daily_before = aggregate_daily_hours(df, glare_occurs_before)
    glare_daily_after = aggregate_daily_hours(df, glare_occurs_after)
    daylight_daily_after = aggregate_daily_hours(df, daylight_ok_after)

    if mode == 1:
        fig, axes = plot_glare_before_after(glare_daily_before, glare_daily_after, year)
    else:
        fig, axes = plot_calendar_heatmaps(glare_daily_after, daylight_daily_after, year)

    print("图已生成；如需导出请自行使用 fig.savefig(...)")
    plt.show()
    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2],
        default=2,
        help="1: 上下子图为改造前后眩光对比；2: 改造后眩光/采光",
    )
    args = parser.parse_args()
    run_calendar_plot(args.mode)
