import os
import time
from itertools import product

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


def get_window_areas():
    A_facade_S, A_facade_N = Config.L * Config.H, Config.L * Config.H
    A_facade_EW = Config.W * Config.H
    A_win_S = A_facade_S * Config.WWR_S
    A_win_N = A_facade_N * Config.WWR_N
    A_win_E = A_facade_EW * Config.WWR_EW
    A_win_W = A_facade_EW * Config.WWR_EW
    return A_win_S, A_win_N, A_win_E, A_win_W


def build_facade_info(flags):
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


def format_facade_label(flags):
    def code(v):
        has_oh, has_fin = v
        if has_oh and has_fin:
            return "OF"
        if has_oh:
            return "O"
        if has_fin:
            return "F"
        return "0"

    return f"S:{code(flags['S'])}|N:{code(flags['N'])}|EW:{code(flags['E'])}"


def count_components(flags):
    s_oh, s_fin = flags["S"]
    n_oh, n_fin = flags["N"]
    e_oh, e_fin = flags["E"]
    w_oh, w_fin = flags["W"]
    return int(s_oh) + int(s_fin) + int(n_oh) + int(n_fin) + int(e_oh) + int(e_fin) + int(w_oh) + int(w_fin)


def evaluate_design_with_facade(design_params, weather_data, sun_vecs, work_mask, facade_info):
    D_oh, D_fin, beta, SHGC_s, SHGC_other = design_params

    A_win_S, A_win_N, A_win_E, A_win_W = get_window_areas()
    total_real_win_area = A_win_S + A_win_N + A_win_E + A_win_W

    Q_solar_total = np.zeros(len(weather_data))
    glare_mask_year = np.zeros(len(weather_data), dtype=bool)
    E_in_diff = np.zeros(len(weather_data))

    for (A_win, az, has_oh, use_fin, is_south) in facade_info:
        SHGC_val = SHGC_s if is_south else SHGC_other
        tau_glass_diff_curr = SHGC_val
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
        Q_s = A_win * (I_trans_dir + SHGC_val * k_diff_dyn * I_diff_outdoor)
        Q_solar_total += Q_s

        is_glare = (
            work_mask
            & (depth > Config.glare_depth_limit)
            & (I_trans_dir > 50.0)
            & (eta > 0.05)
        )
        glare_mask_year |= is_glare

        E_in_diff += (weather_data["DHI"].values * k_diff_dyn * A_win * tau_glass_diff_curr)

    total_wall_area = 2 * (Config.L + Config.W) * Config.H
    UA = ((total_wall_area - total_real_win_area) * Config.U_wall) + (total_real_win_area * Config.U_win)

    Q_int = np.where(work_mask, Config.Q_int_work, Config.Q_int_off)
    Q_load_cool = Q_solar_total + UA * (weather_data["Tout"].values - Config.T_cool) + Q_int
    Q_load_heat = Q_solar_total + UA * (weather_data["Tout"].values - Config.T_heat) + Q_int

    E_cool = np.sum(np.maximum(Q_load_cool, 0)) / Config.COP_cool / 1000.0
    E_heat = np.sum(np.maximum(-Q_load_heat, 0)) / Config.eta_heat / 1000.0
    peak_load = np.max(np.maximum(Q_load_cool, 0)) / 1000.0

    floor_area = Config.L * Config.W * 2
    E_in_diff = (E_in_diff / floor_area) * Config.kappa_lux

    daylight_ok_mask = (E_in_diff >= Config.lux_min) & work_mask
    daylight_hours = np.sum(daylight_ok_mask)
    daylight_ratio = daylight_hours / (np.sum(work_mask) + 1e-9)

    return {
        "J": Config.w_cool * E_cool + Config.w_heat * E_heat,
        "E_cool": E_cool,
        "E_heat": E_heat,
        "Peak": peak_load,
        "GlareHours": np.sum(glare_mask_year),
        "DaylightHours": daylight_hours,
        "DaylightRatio": daylight_ratio,
        "Feasible": (daylight_ratio >= Config.daylight_passing_rate)
        and (np.sum(glare_mask_year) <= Config.glare_hours_max),
    }


def run_config_search():
    weather_path = r"E:\数模\26美赛\数据\weather_Miami Intl Ap.csv"
    df, sun_vecs, work_mask = load_and_preprocess(weather_path, lat_deg=25)

    states = {
        "O": (True, False),
        "F": (False, True),
        "OF": (True, True),
    }

    config_flags = []
    for s_state in states.values():
        for n_state in states.values():
            for ew_state in states.values():
                flags = {
                    "S": s_state,
                    "N": n_state,
                    "E": ew_state,
                    "W": ew_state,
                }
                config_flags.append(flags)

    param_grid = list(
        product(
            Config.D_oh_range,
            Config.D_fin_range,
            Config.Beta_range,
            Config.SHGC_S_range,
            Config.SHGC_Other_range,
        )
    )

    results = []
    start_time = time.time()

    for idx, flags in enumerate(config_flags):
        facade_info = build_facade_info(flags)
        best_any = None
        best_feasible = None

        for params in param_grid:
            res = evaluate_design_with_facade(params, df, sun_vecs, work_mask, facade_info)
            res["params"] = params
            if best_any is None or res["J"] < best_any["J"]:
                best_any = res
            if res["Feasible"]:
                if best_feasible is None or res["J"] < best_feasible["J"]:
                    best_feasible = res

        best = best_feasible if best_feasible is not None else best_any
        s_oh, s_fin = flags["S"]
        n_oh, n_fin = flags["N"]
        e_oh, e_fin = flags["E"]
        w_oh, w_fin = flags["W"]

        results.append(
            {
                "ConfigLabel": format_facade_label(flags),
                "S_oh": int(s_oh),
                "S_fin": int(s_fin),
                "N_oh": int(n_oh),
                "N_fin": int(n_fin),
                "E_oh": int(e_oh),
                "E_fin": int(e_fin),
                "W_oh": int(w_oh),
                "W_fin": int(w_fin),
                "ComponentCount": count_components(flags),
                "J": best["J"],
                "GlareHours": best["GlareHours"],
                "DaylightRatio": best["DaylightRatio"],
                "Feasible": best["Feasible"],
                "Params": best["params"],
            }
        )

        if (idx + 1) % 3 == 0:
            print(f"Config {idx + 1}/{len(config_flags)} done.")

    print(f"All configs done in {time.time() - start_time:.2f}s")

    res_df = pd.DataFrame(results)
    output_dir = os.path.join(os.path.dirname(__file__), "output_config_combo")
    os.makedirs(output_dir, exist_ok=True)

    res_df.sort_values("J", inplace=True)
    res_df.to_csv(os.path.join(output_dir, "config_combo_results.csv"), index=False)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=res_df,
        x="GlareHours",
        y="J",
        hue="DaylightRatio",
        size="ComponentCount",
        sizes=(60, 260),
        palette="viridis",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        legend="brief",
    )
    ax.set_title("Config Combo Scatter (best per config)")
    ax.set_xlabel("GlareHours")
    ax.set_ylabel("J")
    fig.tight_layout()

    print(f"Saved results to: {output_dir}")
    print("Figure created. Use plt.savefig(...) if you want to export.")
    return res_df, fig, ax


if __name__ == "__main__":
    run_config_search()
