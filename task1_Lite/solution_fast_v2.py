import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from itertools import product

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
@dataclass(frozen=True)
class Config:
    # 建筑尺寸
    L, W, H = 60.0, 24.0, 8.0

    # 窗墙比 (WWR)
    WWR_S = 0.45
    WWR_N = 0.30
    WWR_EW = 0.30

    # 窗户几何 (用于计算遮挡比例)
    win_width = 3.0
    win_height = 2.0
    win_sill_h = 1.0

    # 热工参数
    U_wall = 0.6
    U_win = 2.4   # 假设换玻璃只改变 SHGC，不改变 U值 (或者你可以设为联动)

    # --- 决策变量范围 (在此定义) ---
    # 1. 挑檐 (仅南向)
    D_oh_range = np.linspace(0.0, 2.0, 6) # 0, 0.4, ... 2.0
    # 2. 垂直鳍片 (全向)
    D_fin_range = np.linspace(0.0, 0.8, 5) # 0, 0.2, ... 0.8
    # 3. 鳍片角度
    Beta_range = np.linspace(0.0, 60.0, 4) # 0, 20, 40, 60
    # 4. 玻璃 SHGC (新增决策变量)
    # 控制更精细，朝南和其他朝向可使用不同SHGC（建筑可采用两种玻璃）
    # 范围通常在 0.25 (高性能Low-E) 到 0.75 (普通透明) 之间
    SHGC_S_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    SHGC_Other_range = [0.3, 0.4, 0.5, 0.6, 0.7]

    # 系统效率
    COP_cool = 3.2
    eta_heat = 0.9
    T_cool, T_heat = 24.0, 20.0
    Q_int_work = 18000.0
    Q_int_off = 3000.0

    # 约束条件
    glare_depth_limit = 2
    glare_hours_max = 500.0
    lux_min = 300.0
    daylight_passing_rate = 0.5

    # 采光计算参数
    kappa_lux = 100.0

    # 优化权重
    w_cool = 1.0
    w_heat = 1.0

# ==========================================
# 2. 核心计算核 (Vectorized Kernels)
# ==========================================
# (这部分代码与之前完全相同，为节省篇幅省略，请保留原有的 solar_position_numpy,
# get_sun_vectors, calc_shading_factors_fast, calc_diffuse_factor)
def solar_position_numpy(day_of_year, hour, lat_deg):
    """
    向量化计算太阳高度角和方位角 (ENU坐标: x=E, y=N, z=U)
    """
    lat = np.deg2rad(lat_deg)
    declination = np.deg2rad(23.45 * np.sin(np.deg2rad(360.0 * (284.0 + day_of_year) / 365.0)))
    hour_angle = np.deg2rad(15.0 * (hour - 12.0))

    sin_alt = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))

    # 方位角计算
    cos_az = (np.sin(alt) * np.sin(lat) - np.sin(declination)) / (np.cos(alt) * np.cos(lat) + 1e-9)
    az = np.arccos(np.clip(cos_az, -1.0, 1.0))
    # 修正下午时段
    az = np.where(hour_angle > 0, 2 * np.pi - az, az)

    return np.rad2deg(alt), np.rad2deg(az)

def get_sun_vectors(alt_deg, az_deg):
    """返回指向太阳的单位向量 (N, 3)"""
    alt = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)
    z = np.sin(alt)
    r = np.cos(alt)
    x = r * np.sin(az) # East
    y = r * np.cos(az) # North
    return np.stack([x, y, z], axis=1)

def calc_shading_factors_fast(sun_vecs, facade_azimuth_deg, D_oh, D_fin, beta_fin_deg, win_w, win_h):
    """
    解析法计算遮阳系数 eta (直射光透过率)。
    sun_vecs: (T, 3) 世界坐标系下的太阳向量
    facade_azimuth: 立面朝向 (0=N, 90=E, 180=S, 270=W)

    返回:
    eta: (T,) 直射透过比例 [0, 1]
    depth_max: (T,) 最大射入深度 (m)
    """
    # 1. 转换太阳向量到窗户局部坐标系 (u=右/水平, v=上/垂直, n=外/法线)
    # 世界系: x=E, y=N, z=U
    f_az = np.deg2rad(facade_azimuth_deg)
    n_vec = np.array([np.sin(f_az), np.cos(f_az), 0.0]) # 法线
    v_vec = np.array([0.0, 0.0, 1.0])                   # 向上
    u_vec = np.cross(v_vec, n_vec)                      # 向右 (从室外看)

    # 投影
    s_u = np.dot(sun_vecs, u_vec)
    s_v = np.dot(sun_vecs, v_vec) # z分量
    s_n = np.dot(sun_vecs, n_vec)

    # 过滤背对太阳的时刻
    front_mask = s_n > 0.01

    # Profile Angle (用于挑檐计算) -> 投影在垂直面上的夹角
    # tan(phi) = s_v / s_n
    tan_phi = np.zeros_like(s_v)
    tan_phi[front_mask] = s_v[front_mask] / s_n[front_mask]

    # Horizontal Angle (用于垂直鳍片) -> 投影在水平面上的夹角
    # tan(gamma) = s_u / s_n
    tan_gamma = np.zeros_like(s_u)
    tan_gamma[front_mask] = s_u[front_mask] / s_n[front_mask]

    # === A. 挑檐遮挡 (Overhang) ===
    # 阴影从窗顶向下延伸长度 h_shadow = D_oh * tan(phi)
    # 因为太阳在天上，s_v > 0，tan_phi > 0。
    # 只有当太阳很高时，影子才长。
    h_shadow = D_oh * tan_phi
    # 挑檐的透过率 = (窗高 - 阴影长度) / 窗高，截断在 [0, 1]
    frac_oh = np.clip((win_h - h_shadow) / win_h, 0.0, 1.0)

    # === B. 垂直鳍片遮挡 (Vertical Fins) ===
    # 假设鳍片均匀分布。简化模型：使用“百叶窗几何”近似。
    # 这里的 D_fin 是鳍片深度，beta 是旋转角。
    # 我们用“几何截断”法。
    # 有效间距 spacing (假设窗宽分3格)
    spacing = win_w / 3.0
    beta_rad = np.deg2rad(beta_fin_deg)

    # 计算光线通过鳍片阵列的透过率
    # 光线与法线夹角 gamma (在水平面上) = arctan(tan_gamma)
    # 临界角 cutoff：当投影长度 > 间距时完全遮挡
    # 这是一个简化的一维光栅公式
    # 投影的鳍片有效宽度 = D_fin * sin(|gamma - beta|) ? 略复杂
    # 使用更稳健的投影法：
    # 鳍片在光线方向上的投影长度 L_proj = D_fin * |sin(gamma - beta)| / cos(gamma)
    # 但更简单的是看“开口率”。

    # 采用标准百叶窗遮挡公式 (Shading Fraction)
    gamma = np.arctan(tan_gamma) # 太阳水平方位角相对法线的偏角
    # 相对鳍片的入射角
    angle_inc = gamma - beta_rad

    # 鳍片产生的阴影在窗户平面上的水平长度 w_shadow
    # w_shadow = D_fin * sin(gamma - beta) / cos(gamma)  (几何推导)
    # 实际上由于 s_n = cos(gamma), s_u = sin(gamma)，
    # w_shadow = D_fin * (s_u * cos_beta - s_n * sin_beta) / s_n
    #          = D_fin * (tan_gamma * cos_beta - sin_beta)
    # 取绝对值
    w_shadow = np.abs(D_fin * (tan_gamma * np.cos(beta_rad) - np.sin(beta_rad)))

    # 透过率 = (间距 - 阴影) / 间距
    frac_fin = np.clip((spacing - w_shadow) / spacing, 0.0, 1.0)

    # 综合透过率 (假设两个遮挡独立叠加)
    eta = frac_oh * frac_fin
    eta[~front_mask] = 0.0

    # === C. 眩光深度计算 (Glare Depth) ===
    # 深度 d = h_lit / tan(altitude_projected)
    # tan(alt_proj) = s_v / s_n = tan_phi

    # 1. 确定窗户上“最高”的受光点高度 (相对窗底)
    # 挑檐会切掉上半部分，所以最高受光点是 win_h - h_shadow
    h_top_lit = np.clip(win_h - h_shadow, 0.0, win_h)

    # 2. 如果鳍片完全遮挡 (frac_fin = 0)，则受光点高度有效值为 0
    h_top_lit = np.where(frac_fin <= 1e-3, 0.0, h_top_lit)

    # 3. 计算射入深度 (投影到地面)
    # 几何关系：z / depth = tan_phi
    # depth = (win_sill + h_top_lit) / tan_phi
    # 注意：我们要的是在工作面高度? 题目通常指在地板上的投影长度
    # 但要减去窗台高度的影响吗？通常 glare depth 指光斑离墙的最远距离
    # 墙角处 x=0. 光斑最远点 x_max.
    # 光线从 (h_top_lit + sill) 射入。

    # 修正除零
    denom = tan_phi + 1e-9
    depth_max = (Config.win_sill_h + h_top_lit) / denom

    # 如果背对太阳或完全被挡，深度为0
    depth_max[~front_mask] = 0.0
    depth_max[eta < 1e-3] = 0.0

    return eta, depth_max

def calc_diffuse_factor(D_oh, D_fin, beta_fin_deg, win_w, win_h):
    """
    计算动态散射光折减系数 (Sky View Factor 近似)
    解决“完全闭合却还有光”的问题
    """
    # 1. 挑檐造成的视野遮挡近似
    # 视野因子近似为 (1 - 固体角占比)
    # 简单线性近似：D_oh 越大，天空越小
    k_oh = 1.0 - 0.5 * (D_oh / (D_oh + win_h))

    # 2. 鳍片造成的视野遮挡
    # 鳍片完全闭合 (beta=90, D_fin >= spacing) 时，k 应趋近于 0
    spacing = win_w / 3.0
    # 鳍片在垂直于窗面方向的投影占比 (闭合度)
    # projected_closure = (D_fin * |sin(beta)|) / spacing
    closure = (D_fin * np.abs(np.sin(np.deg2rad(beta_fin_deg)))) / spacing
    k_fin = 1.0 - np.clip(closure, 0.0, 1.0) * 0.9 # 即使全关也留10%缝隙漏光，或者设为1.0完全遮挡

    # 综合系数
    k_total = k_oh * k_fin
    return k_total

# ==========================================
# 3. 数据加载与预处理
# ==========================================

def load_and_preprocess(weather_path, lat_deg):
    df = pd.read_csv(weather_path)
    # 确保列名正确
    if 'Tout' not in df.columns: df.rename(columns={'Temperature': 'Tout'}, inplace=True)

    # 解析时间
    df['dt'] = pd.to_datetime(df['datetime'])
    df['doy'] = df['dt'].dt.dayofyear
    df['hour'] = df['dt'].dt.hour + df['dt'].dt.minute / 60.0

    # 预计算太阳向量
    alt, az = solar_position_numpy(df['doy'].values, df['hour'].values, lat_deg)
    sun_vecs = get_sun_vectors(alt, az)

    # 标记工作时间
    work_mask = (df['hour'] >= 8) & (df['hour'] <= 17).values

    return df, sun_vecs, work_mask

# ==========================================
# 4. 评估函数 (支持 SHGC 变量)
# ==========================================

def evaluate_design(design_params, weather_data, sun_vecs, work_mask):
    """
    design_params: (D_oh, D_fin, beta, SHGC_S, SHGC_Other)
    """
    # 1. 解包变量
    D_oh, D_fin, beta, SHGC_s, SHGC_other = design_params

    # 2. 确定物理关联
    # 假设玻璃的散射透射率与 SHGC 成正比
    # 简单模型: tau_diff = SHGC

    # --- 面积计算 ---
    A_facade_S, A_facade_N = Config.L * Config.H, Config.L * Config.H
    A_facade_EW = Config.W * Config.H
    A_win_S = A_facade_S * Config.WWR_S
    A_win_N = A_facade_N * Config.WWR_N
    A_win_E = A_facade_EW * Config.WWR_EW
    A_win_W = A_facade_EW * Config.WWR_EW
    total_real_win_area = A_win_S + A_win_N + A_win_E + A_win_W

    # --- 遮阳计算 ---
    k_diff_dyn = calc_diffuse_factor(D_oh, D_fin, beta, Config.win_width, Config.win_height)

    # 立面配置 (面积, 方位, 是否有挑檐, 是否有侧翼, 是否南向)
    # 策略：北向窗户不做垂直遮阳(D_fin=0)，只受 SHGC 影响
    facade_info = [
        (A_win_S, 180, True, True,  True),   # South
        (A_win_N, 0,   False, False, False), # North (无遮阳构件)
        (A_win_E, 90,  False, True,  False), # East
        (A_win_W, 270, False, True,  False)  # West
    ]

    Q_solar_total = np.zeros(len(weather_data))
    glare_mask_year = np.zeros(len(weather_data), dtype=bool)
    E_in_diff = np.zeros(len(weather_data))

    for (A_win, az, has_oh, use_fin, is_south) in facade_info:
        SHGC_val = SHGC_s if is_south else SHGC_other
        tau_glass_diff_curr = SHGC_val
        d_oh_curr = D_oh if has_oh else 0.0
        d_fin_curr = D_fin if use_fin else 0.0
        beta_curr = beta if use_fin else 0.0

        # 几何计算
        eta, depth = calc_shading_factors_fast(
            sun_vecs, az, d_oh_curr, d_fin_curr, beta_curr,
            Config.win_width, Config.win_height
        )

        # 辐射计算
        f_rad = np.deg2rad(az)
        cos_inc = sun_vecs[:,0]*np.sin(f_rad) + sun_vecs[:,1]*np.cos(f_rad)
        I_dir_outdoor = weather_data['DNI'].values * np.maximum(0, cos_inc)
        I_diff_outdoor = weather_data['DHI'].values * 0.5

        # 关键修改：进入室内的能量受 SHGC 影响
        # I_trans = I_outdoor * eta * SHGC
        I_trans_dir = I_dir_outdoor * eta * SHGC_val

        # 太阳得热
        Q_s = A_win * (I_trans_dir + SHGC_val * k_diff_dyn * I_diff_outdoor)
        Q_solar_total += Q_s

        # 关键修改：眩光判定受 SHGC 影响
        # 如果 SHGC 很低 (如0.25)，即使有光斑，能量也不强，可能不算眩光
        # 设定阈值 50 W/m2
        is_glare = (
            work_mask &
            (depth > Config.glare_depth_limit) &
            (I_trans_dir > 50.0) & # 强度判定包含 SHGC 影响
            (eta > 0.05)
        )
        glare_mask_year |= is_glare

        # 采光累积（分朝向的玻璃透射差异）
        E_in_diff += (weather_data['DHI'].values * k_diff_dyn * A_win * tau_glass_diff_curr)

    # --- 热负荷 ---
    total_wall_area = 2*(Config.L + Config.W)*Config.H
    UA = ((total_wall_area - total_real_win_area) * Config.U_wall) + (total_real_win_area * Config.U_win)

    Q_int = np.where(work_mask, Config.Q_int_work, Config.Q_int_off)
    Q_load_cool = Q_solar_total + UA*(weather_data['Tout'].values - Config.T_cool) + Q_int
    Q_load_heat = Q_solar_total + UA*(weather_data['Tout'].values - Config.T_heat) + Q_int

    E_cool = np.sum(np.maximum(Q_load_cool, 0)) / Config.COP_cool / 1000.0
    E_heat = np.sum(np.maximum(-Q_load_heat, 0)) / Config.eta_heat / 1000.0
    peak_load = np.max(np.maximum(Q_load_cool, 0)) / 1000.0

    # --- 采光 ---
    # 室内平均照度 = 总通量 / 地板面积 * 光效系数
    # 透射光受各朝向 SHGC 影响（已在立面循环中累积 E_in_diff）
    floor_area = Config.L * Config.W * 2
    E_in_diff = (E_in_diff / floor_area) * Config.kappa_lux

    daylight_ok_mask = (E_in_diff >= Config.lux_min) & work_mask
    daylight_hours = np.sum(daylight_ok_mask)
    daylight_ratio = daylight_hours / (np.sum(work_mask) + 1e-9)

    return {
        'J': Config.w_cool * E_cool + Config.w_heat * E_heat,
        'E_cool': E_cool,
        'E_heat': E_heat,
        'Peak': peak_load,
        'GlareHours': np.sum(glare_mask_year),
        'DaylightHours': daylight_hours,
        'DaylightRatio': daylight_ratio,
        'Feasible': (daylight_ratio >= Config.daylight_passing_rate) and (np.sum(glare_mask_year) <= Config.glare_hours_max)
    }

# ==========================================
# 5. 主程序
# ==========================================

def run_optimization():
    print("正在初始化数据...")
    weather_path = r"E:\数模\26美赛\数据\weather_Miami Intl Ap.csv"
    # 请确保此处的纬度正确
    df, sun_vecs, work_mask = load_and_preprocess(weather_path, lat_deg=25)

    # 计算 Baseline (SHGC=0.6, 无遮阳)
    print("计算 Baseline (SHGC=0.6)...")
    base_res = evaluate_design((0.0, 0.0, 0.0, 0.6, 0.6), df, sun_vecs, work_mask)
    print(f"Baseline Energy: {base_res['J']:.0f}, Glare: {base_res['GlareHours']:.0f}h")

    # 网格搜索
    combinations = list(product(
        Config.D_oh_range,
        Config.D_fin_range,
        Config.Beta_range,
        Config.SHGC_S_range,
        Config.SHGC_Other_range
    ))

    print(f"开始搜索 {len(combinations)} 种组合 (含SHGC优化)...")
    start_time = time.time()

    results = []
    for i, params in enumerate(combinations):
        res = evaluate_design(params, df, sun_vecs, work_mask)
        res['params'] = params
        results.append(res)
        if i % 100 == 0: print(f"\rProgress: {i}/{len(combinations)}", end="")

    print(f"\n搜索完成，耗时: {time.time() - start_time:.2f}s")

    # 结果分析
    res_df = pd.DataFrame(results)
    feasible_df = res_df[res_df['Feasible'] == True].copy()

    if len(feasible_df) == 0:
        print("\n" + "!"*40)
        print("警告：没有找到完全满足约束的解！")
        print("正在展示【最接近】的可行解（最小眩光、最大采光、最小能耗）...")

        best_glare = res_df.sort_values('GlareHours').iloc[0]
        best_daylight = res_df.sort_values('DaylightHours', ascending=False).iloc[0]
        best_energy = res_df.sort_values('J').iloc[0]

        def fmt_params(row):
            return (
                f"SHGC_S={row['params'][3]:.2f}, "
                f"SHGC_O={row['params'][4]:.2f}, "
                f"D_oh={row['params'][0]:.2f}, "
                f"D_fin={row['params'][1]:.2f}, "
                f"Beta={row['params'][2]:.1f}"
            )

        print("\n--- 调试信息：无解时的最优候选 ---")
        print(
            f"1. 最小眩光解: Glare={best_glare['GlareHours']:.1f}h, "
            f"DaylightRatio={best_glare['DaylightRatio']:.2f}, "
            f"J={best_glare['J']:.0f}, {fmt_params(best_glare)}"
        )
        print(
            f"2. 最大采光解: Glare={best_daylight['GlareHours']:.1f}h, "
            f"DaylightRatio={best_daylight['DaylightRatio']:.2f}, "
            f"J={best_daylight['J']:.0f}, {fmt_params(best_daylight)}"
        )
        print(
            f"3. 节能最优解: Glare={best_energy['GlareHours']:.1f}h, "
            f"DaylightRatio={best_energy['DaylightRatio']:.2f}, "
            f"J={best_energy['J']:.0f}, {fmt_params(best_energy)}"
        )
        print("!"*40 + "\n")

        best_row = best_energy
    else:
        best_row = feasible_df.loc[feasible_df['J'].idxmin()]

    aesr = (base_res['J'] - best_row['J']) / base_res['J'] * 100

    print("\n" + "="*40)
    print("   OPTIMAL RETROFIT STRATEGY (v2)   ")
    print("="*40)
    print(f"Design Parameters:")
    print(f"  Glass SHGC (South): {best_row['params'][3]:.2f}")
    print(f"  Glass SHGC (Other): {best_row['params'][4]:.2f}")
    print(f"  Overhang Depth (S): {best_row['params'][0]:.2f} m")
    print(f"  Fin Width (E/W/S):  {best_row['params'][1]:.2f} m")
    print(f"  Fin Angle:          {best_row['params'][2]:.1f} deg")
    print("-" * 40)
    print(f"Performance:")
    print(f"  Energy Score (J):     {best_row['J']:.0f}")
    print(f"  Savings (AESR):       {aesr:.2f}%")
    print(f"  Glare Hours:          {best_row['GlareHours']:.1f} h")
    print(f"  Daylight Ratio:       {best_row['DaylightRatio']*100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    run_optimization()
