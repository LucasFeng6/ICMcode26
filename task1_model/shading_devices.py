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
    depth: float          # D_fin (m)  # 中文：侧翼外挑深度 D_fin（米）
    angle_deg: float      # beta_fin (deg) rotation around vertical axis  # 中文：侧翼绕竖直轴旋转角 beta_fin（度）
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
    Two fins at left (u=0) and right (u=win_width) edges.
    Each fin is a thin box with extents [thickness, height, depth] in [u,v,n].
    Then rotate around vertical axis by angle_deg (positive rotates fin plane).

    中文：
    在窗户左右两侧（u=0 与 u=win_width）各放置一个侧翼。
    每个侧翼是一个薄盒体，在 [u,v,n] 方向的尺寸为 [thickness, height, depth]，
    然后绕竖直轴按 angle_deg 旋转（正角度表示侧翼平面按约定方向旋转）。
    """
    if p.depth <= 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)

    fin_h = (win_top_v - win_bottom_v) + p.top_margin + p.bottom_margin
    ext = np.array([p.thickness, fin_h, p.depth], float)
    base = _box(ext)

    # Left fin center (slightly outside edge)  # 中文：左侧翼中心（略微在窗边之外）
    v_center = (win_bottom_v - p.bottom_margin) + fin_h/2.0
    n_center = p.depth/2.0
    left_u_center = -p.side_offset - p.thickness/2.0
    right_u_center = win_width + p.side_offset + p.thickness/2.0

    left = base.copy()
    left.apply_translation([left_u_center, v_center, n_center])

    right = base.copy()
    right.apply_translation([right_u_center, v_center, n_center])

    # Rotate about vertical axis through each fin's center line at u edge  # 中文：绕竖直轴旋转，轴线通过各侧翼在 u 边界处的中心线
    # Pivot points chosen at fin centers to keep simple (can refine later)  # 中文：为简化，选取侧翼中心作为旋转枢轴点（可进一步精细化）
    left = _rotate_about_v(left, p.angle_deg, pivot=np.array([left_u_center, v_center, 0.0]))
    right = _rotate_about_v(right, -p.angle_deg, pivot=np.array([right_u_center, v_center, 0.0]))

    combo = trimesh.util.concatenate([left, right])
    return combo
