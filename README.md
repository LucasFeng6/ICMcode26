## 项目简介

本仓库提供 ICM/数模 Task 1 的一套**可复现**计算流程：在给定气象数据（`Tout/DNI/DHI`）与建筑/窗户参数的前提下，
对“基准方案（无遮阳）”与“改造方案（被动遮阳：南向挑檐 + 其余立面旋转鳍片）”进行评估，并通过网格搜索找到满足舒适性约束且能耗目标最优的遮阳参数组合。

> 备注：代码中的“眩光/采光”是题目驱动的简化代理模型，不等同于 DGP/UGR 等标准指标，详见下文“模型假设与限制”。

---

## 目录结构（快速定位）

- `task1_model/config.py`：核心参数配置（建筑几何 / 围护结构 / HVAC / 内部得热 / 采光 / 舒适性约束 / 权重）
- `task1_model/main_task1.py`：**主入口**（读取天气 → 基准评估 → 网格搜索优化）
- `task1_model/evaluate.py`：单个方案评估（遮阳→直射受光比例→眩光/采光→能耗→指标）
- `task1_model/optimize.py`：网格搜索（带缓存的加速版）
- `task1_model/shading_devices.py`：遮阳几何（南向挑檐、N/E/W 三片旋转鳍片）
- `task1_model/comfort.py`：眩光与采光判定（代理模型）
- `task1_model/thermal.py`：简化热负荷/能耗模型
- `export_baseline_comfort.py`：导出基准方案的“眩光深度/采光照度”时间序列到 CSV（便于画图/检查阈值）

---

## 环境与依赖

- Python：建议 `>= 3.10`（代码使用了 `list[...]` 等类型注解写法）
- 依赖包（最小集合）：
  - `numpy`
  - `pandas`
  - `trimesh`

安装示例（PowerShell，在项目根目录执行）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install numpy pandas trimesh
```

---

## 输入数据：天气 CSV 规范

所有主流程均基于天气 CSV（例如逐小时）：

- 必需列（大小写需一致）：
  - `datetime`：可被 `pandas.to_datetime` 解析的时间字符串（建议 ISO）
  - `Tout`：室外温度（℃）
  - `DNI`：法向直射辐照（W/m²）
  - `DHI`：水平散射辐照（W/m²）
- 时间步长：代码通过相邻两行 `datetime` 推断 `dt_hours`（小时）。

---

## 推荐工作流程（从 0 到出结果）

### Step 0：准备天气文件

将天气 CSV 放到任意位置，并确保列名满足上面的规范。

### Step 1（可选但推荐）：导出基准舒适性时间序列（便于做 sanity check）

用于快速检查“眩光阈值/采光阈值”在基准方案下的时间分布，并生成可画图的 CSV。

```powershell
python export_baseline_comfort.py --weather "E:\path\weather.csv" --out "baseline_comfort.csv" --lat 25.8
```

输出 `baseline_comfort.csv` 字段说明（节选）：

- `glare_depth_[N/E/S/W]_m`：各立面在每个时刻的眩光“深度”代理值（米）
- `glare_depth_max_m`：四个立面的最大眩光深度（米）
- `glare_flag`：工作时段内是否超过阈值（0/1）
- `E_in_lux`：室内平均照度代理值（lux，仅由 DHI 推得）
- `daylight_ok_t`：该时刻是否满足采光阈值（非工作时段恒为 1）

### Step 2：运行 Task1 主流程（基准评估 + 网格优化）

主入口在 `task1_model/main_task1.py`。首次接手项目建议按顺序做两件事：

1) 打开 `task1_model/main_task1.py`，修改用户输入区：
   - `weather_path`：你的天气 CSV 路径
   - `lat_deg`：纬度（度）
   - `wins`：四个立面的代表性窗户几何（`width/height/sill_h/floor_z`）
   - `nx, ny`：窗面采样网格分辨率

2) 在项目根目录运行（推荐用 `-m` 方式）：

```powershell
python -m task1_model.main_task1
```

终端输出包含两部分：

- `=== Baseline ===`：无遮阳基准方案（用于后续计算 AESR/PLR）
- `=== Best feasible retrofit ===`：满足约束的最优遮阳方案（网格搜索）

> 注意：`main_task1.py` 末尾打印行里写死了 `"(<= 50)"` 作为示例文字；**实际眩光小时阈值以 `task1_model/config.py` 的 `ComfortConstraints.glare_hours_max` 为准**。

---

## 工作流与数据流（代码级对照）

以 `task1_model/main_task1.py` 为主线：

1) `load_weather_csv()`：读取天气 CSV，得到 `times/weather/dt_hours`
2) `compute_UA_total()`：根据 `BuildingGeom/EnvelopeParams` 计算
   - `UA_total`：围护结构总传热系数×面积（W/K）
   - `Awin`：四向窗面积（m²）
3) `evaluate_design()`（`task1_model/evaluate.py`）：给定遮阳设计 `Design(D_oh, fin_w, beta_fin_deg)`，依次计算：
   - 遮阳几何（`task1_model/shading_devices.py`）→ `RayScene`
   - 窗面采样点（`task1_model/sampling.py`）→ (N,3) local 点
   - 逐时太阳位置与立面入射辐照（`task1_model/solar_geometry.py`）
   - 直射受光比例 `eta(t)`（trimesh 射线投射求遮挡）
   - 舒适性：
     - 眩光小时数（`task1_model/comfort.compute_glare_hours`）
     - 采光合格小时数（`task1_model/comfort.daylight_ok`）
   - 能耗（`task1_model/thermal.compute_energy`）：冷/热能耗、冷峰值、目标函数 `J`
   - 指标：
     - `AESR(%)`：相对基准目标函数的节能率
     - `PLR(%)`：相对基准冷峰值的削减率
4) `grid_search()`（`task1_model/optimize.py`）：对给定参数网格搜索，利用“南向挑檐可分离、N/E/W 鳍片可分离”的结构做缓存加速，返回能耗目标 `J` 最小的可行解。

---

## 参数说明（务必先读）

### A. 全局配置：`task1_model/config.py`

#### 1) `BuildingGeom`（建筑几何与窗墙比）

- `L`：建筑长度（m）
- `W`：建筑宽度（m）
- `H`：建筑总高度（m，默认 2 层合计）
- `wwr_south`：南向窗墙比（0~1）
- `wwr_other`：其余立面窗墙比（0~1）

#### 2) `HVACParams`（HVAC 简化参数）

- `T_cool`：制冷设定温度（℃，当前模型未显式使用 setpoint 逻辑，保留作扩展）
- `T_heat`：供暖设定温度（℃，同上）
- `COP_cool`：制冷 COP（无量纲，用于把冷负荷换算为电耗）
- `eta_heat`：供暖效率（无量纲，用于把热负荷换算为能耗）

#### 3) `EnvelopeParams`（围护结构与透光/得热参数）

- `U_wall`：墙体传热系数（W/m²·K）
- `U_win`：窗体传热系数（W/m²·K）
- `SHGC`：太阳得热系数（无量纲，用于太阳辐照 → 得热）
- `tau_diff`：散射光可见光透射率（无量纲，用于 DHI → 室内照度）

#### 4) `InternalGains`（内部得热）

- `Q_internal_work`：工作时段内部得热（W）
- `Q_internal_off`：非工作时段内部得热（W）

#### 5) `DaylightingParams`（采光模型参数）

采光模型（仅散射光）：

`E_in_avg(t) = C_dl * kappa * DHI(t) * sum( tau_diff * Awin_facade * k_diff ) / A_floor`

- `kappa`：把辐照（W/m²）映射到照度（lux）的比例系数（经验值，可调）
- `C_dl`：从“透射到室内平均照度”的折减系数（经验值，可调）
- `k_diff_shade`：遮阳对散射光的削弱系数（0~1；越小表示遮得越多）

#### 6) `ComfortConstraints`（舒适性约束）

- `glare_depth_m`：眩光“深度”阈值（m）
- `glare_hours_max`：允许的全年眩光小时上限（h）
- `daylight_lux_min`：工作时段室内照度下限（lux）
- `daylight_ok_hours_min`：全年工作时段内“照度合格”的最少小时数（h）
- `work_start_hour`：工作开始小时（包含端点）
- `work_end_hour`：工作结束小时（包含端点）

#### 7) `OptimizationWeights`（目标函数权重）

- `w_cool`：制冷能耗权重
- `w_heat`：供暖能耗权重

目标函数：`J = w_cool * E_cool_kWh + w_heat * E_heat_kWh`（见 `task1_model/thermal.py`）

### B. 入口脚本内参数：`task1_model/main_task1.py`

这些是“随题目/场景变化最大”的参数，通常每次换城市/建筑都要改：

- `weather_path`：天气 CSV 路径
- `lat_deg`：纬度（度）
- `wins`（四向窗几何）：
  - `width`：窗宽（m）
  - `height`：窗高（m）
  - `sill_h`：窗底距该层楼面的高度（m）
  - `floor_z`：该窗所属楼层楼面的世界坐标 `z`（m）
- `nx, ny`：窗面采样网格分辨率（越大越精细但更慢）
- 网格范围（可按题意调整）：
  - `D_oh_list`：南向挑檐外挑深度（m）
  - `fin_w_list`：N/E/W 鳍片宽度（m）
  - `beta_list`：鳍片旋转角（deg，`0` 为完全打开）

### C. 基准舒适性导出脚本：`export_baseline_comfort.py`

常用参数（见脚本内 `--help`）：

- `--weather`：输入天气 CSV（必填）
- `--out`：输出 CSV（默认 `baseline_comfort.csv`）
- `--lat`：纬度（度）
- `--nx/--ny`：窗面采样密度
- `--glare-depth-thresh`：眩光深度阈值（m）
- `--lux-min`：照度阈值（lux）
- `--tau-diff/--kappa/--C-dl`：覆盖 `config.py` 中的采光参数（可用于校准）

---

## 模型假设与限制（接手者需要心里有数）

- 太阳位置：简化太阳高度/方位计算，未引入均时差等更精细项（见 `task1_model/solar_geometry.py`）。
- 眩光：采用“窗面采样点 + 射线遮挡 + 眩光深度代理阈值”的判定逻辑（见 `task1_model/comfort.py`），不是标准眩光指标。
- 采光：仅用 `DHI` 估算室内平均照度，未显式考虑直射采光、室内反射、遮阳几何导致的空间非均匀性等。
- 热模型：单区、线性传热 + 太阳得热 + 内部得热的简化负荷模型；HVAC 以 COP/效率做能耗换算（见 `task1_model/thermal.py`）。
- 遮阳几何：
  - 南向仅使用挑檐（`D_oh`）
  - N/E/W 仅使用三片等宽旋转鳍片（`fin_w`, `beta_fin_deg`），转轴固定在窗面外侧 `n=0.5m`

---

## 常见问题（快速排障）

- 中文注释乱码：本项目源码为 UTF-8，无 BOM。在 PowerShell 5.1 里用某些命令查看可能出现乱码，不影响 Python 运行。
- 运行方式建议：请在项目根目录运行脚本/模块，避免 `import task1_model...` 找不到的路径问题。
- 网格搜索找不到可行解：优先检查
  - `ComfortConstraints` 的阈值是否过严
  - `wins` 窗几何是否与题意一致
  - 网格范围是否覆盖合理的遮阳尺寸
