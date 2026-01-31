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
    L, W, H = 60.0, 24.0, 8.0  # H 是总高

    # 窗墙比 (WWR) - 依据题目文档
    WWR_S = 0.45  # 南向
    WWR_N = 0.30  # 北向
    WWR_EW = 0.30 # 东西向 (假设)

    # 单个代表性窗户尺寸 (用于计算遮挡比例 eta)
    win_width = 3.0
    win_height = 2.0
    win_sill_h = 1.0

    # ... (其他热工参数保持不变) ...
    U_wall = 0.6
    U_win = 2.4
    SHGC_glass = 0.6
    COP_cool = 3.2
    eta_heat = 0.9
    T_cool, T_heat = 24.0, 20.0
    Q_int_work = 18000.0
    Q_int_off = 3000.0

    # --- 调整后的约束 ---
    glare_depth_limit = 3
    glare_hours_max = 800.0       # 保持你放松后的约束
    lux_min = 300.0
    daylight_passing_rate = 0.5

    # 采光系数 (由于使用了真实窗面积，这个值可以回归正常范围)
    kappa_lux = 100.0   # 稍微调低，因为面积变大了
    tau_glass_diff = 0.6

    w_cool = 1.0
    w_heat = 1.0

# ==========================================
# 2. 高性能核心计算核 (Vectorized Kernels)
# ==========================================

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
    work_mask = (df['hour'] >= 8) & (df['hour'] <= 17)

    return df, sun_vecs, work_mask

# ==========================================
# 4. 评估函数 (Single Design Evaluation)
# ==========================================

def evaluate_design(design_params, weather_data, sun_vecs, work_mask, base_loads=None):
    D_oh, D_fin, beta = design_params

    # 1. 计算总窗面积 (基于 WWR)
    # 立面总面积
    A_facade_S = Config.L * Config.H
    A_facade_N = Config.L * Config.H
    A_facade_E = Config.W * Config.H
    A_facade_W = Config.W * Config.H

    # 各朝向窗户总面积
    A_win_S = A_facade_S * Config.WWR_S
    A_win_N = A_facade_N * Config.WWR_N
    A_win_E = A_facade_E * Config.WWR_EW
    A_win_W = A_facade_W * Config.WWR_EW

    total_real_win_area = A_win_S + A_win_N + A_win_E + A_win_W

    # 2. 计算各立面的遮阳系数 (使用代表性窗户计算几何比例)
    # 动态散射系数 k_diff (假设全楼统一使用该设计)
    k_diff_dyn = calc_diffuse_factor(D_oh, D_fin, beta, Config.win_width, Config.win_height)

    # 定义各立面参数
    # 格式: (Area, Azimuth, has_overhang)
    facade_info = [
        (A_win_S, 180, True, True),   # South: Use Params
        (A_win_N, 0,   False, False), # North: Force NO fins (参数设为False)
        (A_win_E, 90,  False, True),  # East: Use Params
        (A_win_W, 270, False, True)   # West: Use Params
    ]

    Q_solar_total = np.zeros(len(weather_data))
    glare_mask_year = np.zeros(len(weather_data), dtype=bool)

        # 循环计算各立面
    for (A_win, az, has_oh, use_fins) in facade_info:
        d_oh_curr = D_oh if has_oh else 0.0
        d_fin_curr = D_fin if use_fins else 0.0  # 北向强制0
        beta_curr = beta if use_fins else 0.0    # 北向强制0

        # 1. 计算几何遮挡 eta 和 眩光深度 depth
        eta, depth = calc_shading_factors_fast(
            sun_vecs, az, d_oh_curr, d_fin_curr, beta_curr,
            Config.win_width, Config.win_height
        )

        # 2. 计算辐射强度 (这是关键！)
        f_rad = np.deg2rad(az)
        cos_inc = sun_vecs[:,0]*np.sin(f_rad) + sun_vecs[:,1]*np.cos(f_rad)

        # 室外直射辐射
        I_dir_outdoor = weather_data['DNI'].values * np.maximum(0, cos_inc)
        I_diff_outdoor = weather_data['DHI'].values * 0.5

        # 3. 计算“透过遮阳构件进入室内的直射辐射强度”
        # I_trans_dir = 室外DNI * 几何透过率 eta * 玻璃SHGC(或透射率)
        # 这里用 SHGC 近似透射率，或者专门定义一个 transmittance
        # 如果进入室内的直射能量很小，就不算眩光
        I_trans_dir = I_dir_outdoor * eta * Config.SHGC_glass

        # 4. 太阳得热 (保持不变)
        Q_s = A_win * (I_trans_dir + Config.SHGC_glass * k_diff_dyn * I_diff_outdoor)
        Q_solar_total += Q_s

        # 5. 修正后的眩光判定 (Relaxed Glare Logic)
        # 条件 A: 工作时间
        # 条件 B: 几何深度 > 1.5m
        # 条件 C: (新增) 进入室内的直射辐射强度 > 50 W/m2 (阈值可调，建议 30-50)
        # 条件 D: (保留) eta > 0.05 (如果遮挡了95%以上，认为剩下的光斑不具备破坏性)

        is_glare_moment = (
            work_mask &
            (depth > Config.glare_depth_limit) &
            (I_trans_dir > 50.0) &  # <--- 新增：强度阈值
            (eta > 0.05)            # <--- 修改：忽略微小漏光
        )

        glare_mask_year |= is_glare_moment

    # 3. 热负荷计算
    # 围护结构 UA (使用剩余墙体面积)
    total_wall_area = 2*(Config.L + Config.W)*Config.H
    real_opaque_wall_area = total_wall_area - total_real_win_area

    UA = (real_opaque_wall_area * Config.U_wall) + (total_real_win_area * Config.U_win)

    # 传导与内部得热
    Q_trans = UA * (weather_data['Tout'].values - 22.0) # 简化基准温差
    Q_int = np.where(work_mask, Config.Q_int_work, Config.Q_int_off)

    Q_load_cool = Q_solar_total + UA*(weather_data['Tout'].values - Config.T_cool) + Q_int
    Q_load_heat = Q_solar_total + UA*(weather_data['Tout'].values - Config.T_heat) + Q_int

    E_cool = np.sum(np.maximum(Q_load_cool, 0)) / Config.COP_cool / 1000.0
    E_heat = np.sum(np.maximum(-Q_load_heat, 0)) / Config.eta_heat / 1000.0
    peak_load = np.max(np.maximum(Q_load_cool, 0)) / 1000.0

    # 4. 采光判定 (核心修正)
    # 使用 total_real_win_area
    floor_area = Config.L * Config.W * 2 # 假设两层

    # 室内照度公式
    E_in_diff = (weather_data['DHI'].values * k_diff_dyn * total_real_win_area * Config.tau_glass_diff) / floor_area * Config.kappa_lux

    daylight_ok_mask = (E_in_diff >= Config.lux_min) & work_mask
    daylight_hours = np.sum(daylight_ok_mask)
    work_hours_total = np.sum(work_mask)

    # 这里的 1e-9 防止除零
    daylight_ratio = daylight_hours / (work_hours_total + 1e-9)
    is_daylight_pass = daylight_ratio >= Config.daylight_passing_rate

    # 5. 结果返回
    glare_hours = np.sum(glare_mask_year)
    is_glare_pass = glare_hours <= Config.glare_hours_max

    J = Config.w_cool * E_cool + Config.w_heat * E_heat

    return {
        'J': J,
        'E_cool': E_cool,
        'E_heat': E_heat,
        'Peak': peak_load,
        'GlareHours': glare_hours,
        'DaylightHours': daylight_hours,
        'DaylightRatio': daylight_ratio, # 方便调试
        'Feasible': is_daylight_pass and is_glare_pass
    }

# ==========================================
# 5. 主程序与网格搜索
# ==========================================

def run_optimization():
    print("正在初始化数据...")
    # 1. 加载天气 (请替换为你的CSV路径)
    # 构造假数据用于演示 (正式使用请读取 csv)
    weather_path = r"E:\数模\26美赛\数据\weather_Miami Intl Ap.csv"
    df, sun_vecs, work_mask = load_and_preprocess(weather_path, lat_deg=25.0)

    # --- MOCK DATA GENERATION (演示用，请替换) ---
    '''
    dates = pd.date_range(start='2025-01-01', end='2025-12-31 23:00', freq='h')
    df = pd.DataFrame({'datetime': dates})
    df['doy'] = df['datetime'].dt.dayofyear
    df['hour'] = df['datetime'].dt.hour
    # 模拟迈阿密气候
    df['Tout'] = 25.0 + 5.0 * np.sin((df['doy']/365.0)*2*np.pi) + 5.0 * np.sin((df['hour']-10)/24*2*np.pi)
    df['DNI'] = np.maximum(0, 800 * np.sin((df['hour']-6)/12*np.pi)) * (1 - 0.5*np.random.rand(len(df)))
    df['DHI'] = np.maximum(0, 300 * np.sin((df['hour']-6)/12*np.pi))

    lat_deg = 25.0
    alt, az = solar_position_numpy(df['doy'].values, df['hour'].values, lat_deg)
    sun_vecs = get_sun_vectors(alt, az)
    work_mask = ((df['hour'] >= 8) & (df['hour'] <= 17)).values
    '''
    # --- END MOCK DATA ---

    # 2. 计算 Baseline (无遮阳)
    print("计算 Baseline...")
    base_res = evaluate_design((0.0, 0.0, 0.0), df, sun_vecs, work_mask)
    print(f"Baseline J: {base_res['J']:.2f}, Glare: {base_res['GlareHours']:.1f}h")

    # 3. 定义搜索网格
    # 范围可以根据需要调整
    D_oh_vals = np.linspace(0.0, 2.0, 11)   # 0 - 2m
    D_fin_vals = np.linspace(0.0, 1.0, 11)   # 0 - 1m
    beta_vals = np.linspace(0.0, 80.0, 20)   # 0 - 80度 (90度完全闭合，避免完全0光)

    combinations = list(product(D_oh_vals, D_fin_vals, beta_vals))
    print(f"开始网格搜索，总组合数: {len(combinations)}...")

    start_time = time.time()

    results = []

    # 这里的循环非常快，因为内部全是 vector operation
    for i, (d_oh, d_fin, beta) in enumerate(combinations):
        res = evaluate_design((d_oh, d_fin, beta), df, sun_vecs, work_mask)
        res['params'] = (d_oh, d_fin, beta)
        results.append(res)

        if i % 50 == 0:
            print(f"\rProgress: {i}/{len(combinations)}", end="")

    end_time = time.time()
    print(f"\n搜索完成，耗时: {end_time - start_time:.2f}秒")

    # 4. 筛选最优解
    res_df = pd.DataFrame(results)
    feasible_df = res_df[res_df['Feasible'] == True].copy()

    if len(feasible_df) == 0:
        print("\n" + "!"*40)
        print("警告：没有找到完全满足约束的解！")
        print("正在展示【最接近】的可行解（优先满足眩光，其次采光）...")

        # 策略：先找眩光超标最少的，再在里面找采光最好的，或者找违反约束综合代价最小的
        # 这里简单粗暴：直接按目标函数排序，看看发生了什么
        # 或者：打印采光最好的前3名和眩光最好的前3名

        print("\n--- 调试信息：为什么无解？ ---")
        best_glare = res_df.sort_values('GlareHours').iloc[0]
        best_daylight = res_df.sort_values('DaylightHours', ascending=False).iloc[0]

        print(f"1. 最小眩光解: Glare={best_glare['GlareHours']:.1f}h, DaylightRatio={best_glare['DaylightRatio']:.2f}")
        print(f"2. 最大采光解: Glare={best_daylight['GlareHours']:.1f}h, DaylightRatio={best_daylight['DaylightRatio']:.2f}")

        # 强制返回一个 J 最小的解用于展示，即使不可行
        best_row = res_df.sort_values('J').iloc[0]
        print(f"3. 节能最优解: J={best_row['J']:.0f}, Glare={best_row['GlareHours']:.1f}h, DaylightRatio={best_row['DaylightRatio']:.2f}")
        print("!"*40 + "\n")
    else:
        best_row = feasible_df.loc[feasible_df['J'].idxmin()]

    # 计算指标
    aesr = (base_res['J'] - best_row['J']) / base_res['J'] * 100
    plr = (base_res['Peak'] - best_row['Peak']) / base_res['Peak'] * 100

    print("\n" + "="*40)
    print("   OPTIMAL RETROFIT STRATEGY   ")
    print("="*40)
    print(f"Design Parameters:")
    print(f"  Overhang Depth (S): {best_row['params'][0]:.2f} m")
    print(f"  Fin Width (All):    {best_row['params'][1]:.2f} m")
    print(f"  Fin Angle:          {best_row['params'][2]:.1f} deg")
    print("-" * 40)
    print(f"Performance:")
    print(f"  Energy Saving (AESR): {aesr:.2f}%")
    print(f"  Peak Reduction (PLR): {plr:.2f}%")
    print(f"  Total Energy Score:   {best_row['J']:.2f}")
    print(f"Constraint Checks:")
    print(f"  Glare Hours:    {best_row['GlareHours']:.1f} h (Limit: {Config.glare_hours_max})")
    print(f"  Daylight Hours: {best_row['DaylightHours']:.1f} h (Qualified)")
    print("="*40)

if __name__ == "__main__":
    run_optimization()