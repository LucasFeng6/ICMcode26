import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
DESIGN_PARAMS = (0.8, 0.4, 20.0, 0.5, 0.5)

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
    glare_ok = work_mask & (~glare_mask_year)

    return glare_ok, daylight_ok


def aggregate_daily_hours(df, mask):
    tmp = df.copy()
    tmp["date"] = tmp["dt"].dt.date
    tmp["ok"] = mask.astype(int)
    return tmp.groupby("date")["ok"].sum()


def build_calendar_matrix(daily_series, year):
    dates = pd.to_datetime(daily_series.index)
    values = daily_series.values

    start = pd.Timestamp(year=year, month=1, day=1)
    week_index = ((dates - start).days + start.weekday()) // 7
    dow = dates.weekday  # 0=Mon

    n_weeks = int(week_index.max()) + 1
    mat = np.full((7, n_weeks), np.nan)
    for d, w, v in zip(dow, week_index, values):
        mat[int(d), int(w)] = v

    return mat, start


def month_tick_positions(start, year):
    positions = []
    labels = []
    for m in range(1, 13):
        d = pd.Timestamp(year=year, month=m, day=1)
        w = ((d - start).days + start.weekday()) // 7
        positions.append(w + 0.5)
        labels.append(d.strftime("%b"))
    return positions, labels


def plot_calendar_heatmaps(glare_daily, daylight_daily, year):
    glare_mat, start = build_calendar_matrix(glare_daily, year)
    daylight_mat, _ = build_calendar_matrix(daylight_daily, year)

    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for ax, mat, title in [
        (axes[0], glare_mat, "Glare Constraint Hours (per day)"),
        (axes[1], daylight_mat, "Daylight Constraint Hours (per day)"),
    ]:
        sns.heatmap(
            mat,
            ax=ax,
            cmap="YlGn",
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            cbar_kws={"label": "Hours"},
        )
        ax.set_title(title)
        ax.set_ylabel("")
        ax.set_yticks(np.arange(0.5, 7.5, 1.0))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)

    pos, labels = month_tick_positions(start, year)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(labels, rotation=0)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(labels, rotation=0)

    fig.tight_layout()
    return fig, axes


def run_calendar_plot():
    df, sun_vecs, work_mask = load_and_preprocess(WEATHER_PATH, lat_deg=LAT_DEG)
    facade_info = build_facade_info(FACADE_FLAGS)
    glare_ok, daylight_ok = compute_hourly_masks(
        DESIGN_PARAMS, df, sun_vecs, work_mask, facade_info
    )

    year = int(df["dt"].dt.year.mode()[0])
    glare_daily = aggregate_daily_hours(df, glare_ok)
    daylight_daily = aggregate_daily_hours(df, daylight_ok)

    fig, axes = plot_calendar_heatmaps(glare_daily, daylight_daily, year)
    print("图已生成；如需导出请自行使用 fig.savefig(...)")
    plt.show()
    return fig, axes, glare_daily, daylight_daily


if __name__ == "__main__":
    run_calendar_plot()
