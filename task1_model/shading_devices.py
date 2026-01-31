# task1_model/shading_devices.py
from dataclasses import dataclass
import numpy as np
import trimesh

@dataclass(frozen=True)
class OverhangParams:
    depth: float          # D_oh (m) outward along +n
    gap_above: float = 0.05   # g (m) above window top
    thickness: float = 0.03   # plate thickness (m)
    side_margin: float = 0.10 # extend beyond window width (m)

@dataclass(frozen=True)
class FinParams:
    depth: float          # D_fin (m)
    angle_deg: float      # beta_fin (deg) rotation around vertical axis
    thickness: float = 0.03
    side_offset: float = 0.00 # offset from window edge
    top_margin: float = 0.05
    bottom_margin: float = 0.05

def _box(extents_xyz: np.ndarray) -> trimesh.Trimesh:
    # trimesh box centered at origin
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
    """
    if p.depth <= 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)

    fin_h = (win_top_v - win_bottom_v) + p.top_margin + p.bottom_margin
    ext = np.array([p.thickness, fin_h, p.depth], float)
    base = _box(ext)

    # Left fin center (slightly outside edge)
    v_center = (win_bottom_v - p.bottom_margin) + fin_h/2.0
    n_center = p.depth/2.0
    left_u_center = -p.side_offset - p.thickness/2.0
    right_u_center = win_width + p.side_offset + p.thickness/2.0

    left = base.copy()
    left.apply_translation([left_u_center, v_center, n_center])

    right = base.copy()
    right.apply_translation([right_u_center, v_center, n_center])

    # Rotate about vertical axis through each fin's center line at u edge
    # Pivot points chosen at fin centers to keep simple (can refine later)
    left = _rotate_about_v(left, p.angle_deg, pivot=np.array([left_u_center, v_center, 0.0]))
    right = _rotate_about_v(right, -p.angle_deg, pivot=np.array([right_u_center, v_center, 0.0]))

    combo = trimesh.util.concatenate([left, right])
    return combo
