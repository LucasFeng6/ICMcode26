# task1_model/shading_devices.py  # 中文：遮阳构件（挑檐/侧翼）几何建模
from dataclasses import dataclass
import numpy as np
import trimesh

@dataclass(frozen=True)
class OverhangParams:
    depth: float          # D_oh (m) outward along +n  # 中文：挑檐外挑深度 D_oh（米），沿 +n（外法线）方向
    gap_above: float = 0.05   # g (m) above window top  # 中文：挑檐与窗顶之间的竖向间隙 g（米）
    thickness: float = 0.03   # plate thickness (m)  # 中文：板厚（米）
    side_margin: float = 0.10 # extend beyond window width (m)  # 中文：左右超出窗宽的边距（米）

@dataclass(frozen=True)
class FinParams:
    depth: float          # fin_w (m) -> fin width (panel width)  # 中文：fin_w 作为鳍片宽度（米），最大不超过窗宽/3
    angle_deg: float      # beta_fin (deg) rotation around vertical axis  # 中文：鳍片绕竖直轴旋转角 beta_fin（度）
    thickness: float = 0.03
    side_offset: float = 0.00 # offset from window edge  # 中文：相对窗边的水平偏移（米）
    top_margin: float = 0.05
    bottom_margin: float = 0.05

def _box(extents_xyz: np.ndarray) -> trimesh.Trimesh:
    # trimesh box centered at origin  # 中文：trimesh 的盒子几何默认以原点为中心
    return trimesh.creation.box(extents=extents_xyz)

def make_overhang_mesh_local(win_width: float, win_top_v: float, p: OverhangParams) -> trimesh.Trimesh:
    """
    Build overhang in window-local coordinates:
      u: width axis
      v: up
      n: outward
    Overhang is a thin box: extents [u, v, n] = [W+2m, thickness, depth]
    centered at:
      u = W/2
      v = win_top + gap + thickness/2
      n = depth/2

    中文：
    在窗户局部坐标系中构建挑檐：
      u：沿窗宽方向；v：向上；n：朝外法线方向。
    挑檐用一个薄盒体表示，其尺寸 [u, v, n] = [W+2m, thickness, depth]，
    并以如下中心位置放置：
      u = W/2；v = 窗顶 + 间隙 + thickness/2；n = depth/2。
    """
    if p.depth <= 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)

    ext = np.array([win_width + 2*p.side_margin, p.thickness, p.depth], float)
    mesh = _box(ext)
    center = np.array([win_width/2.0, win_top_v + p.gap_above + p.thickness/2.0, p.depth/2.0], float)
    mesh.apply_translation(center)
    return mesh

def _rotate_about_v(mesh: trimesh.Trimesh, angle_deg: float, pivot: np.ndarray) -> trimesh.Trimesh:
    ang = np.deg2rad(angle_deg)
    R = trimesh.transformations.rotation_matrix(angle=ang, direction=[0,1,0], point=pivot)
    mesh = mesh.copy()
    mesh.apply_transform(R)
    return mesh

def make_fins_mesh_local(win_width: float, win_bottom_v: float, win_top_v: float, p: FinParams) -> trimesh.Trimesh:
    """
    Three rotating panels across the window width (u direction).
    - fin width is a decision variable (fin_w) with an upper bound of win_width/3.
    - beta_fin_deg = 0 means fully open (panel plane ⟂ window plane).
    - All 3 panels rotate in the same direction.

    中文：
    沿窗宽方向（u）均匀布置 3 片可旋转面板：
    - 鳍片宽度为决策变量 fin_w，但不超过窗宽/3（极限时可完全遮挡窗户）。
    - beta_fin_deg = 0 表示“完全打开”（面板平面与窗面垂直）。
    - 3 片面板同向旋转。
    """
    if p.depth <= 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)

    panel_count = 3
    max_panel_w = win_width / panel_count
    panel_w = min(p.depth, max_panel_w)
    # Pivot axis is placed at n = 0.5m from the window plane (per requirement),
    # so panels (<= 1.0m wide) will not collide with the window.
    pivot_n = 0.5

    fin_h = (win_top_v - win_bottom_v) + p.top_margin + p.bottom_margin
    # Panel is a thin box whose "width" is along local n when beta=0 (open state),
    # so at beta=90 it rotates to be parallel to the window plane and can fully cover it.
    ext = np.array([p.thickness, fin_h, panel_w], float)  # [u, v, n]
    base = _box(ext)

    v_center = (win_bottom_v - p.bottom_margin) + fin_h / 2.0

    panels: list[trimesh.Trimesh] = []
    for i in range(panel_count):
        u_center = (i + 0.5) * panel_w
        panel = base.copy()
        panel.apply_translation([u_center, v_center, pivot_n])
        panel = _rotate_about_v(panel, p.angle_deg, pivot=np.array([u_center, v_center, pivot_n]))
        panels.append(panel)

    return trimesh.util.concatenate(panels)
