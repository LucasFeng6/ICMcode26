# task1_model/sampling.py  # 中文：窗面采样点生成
from dataclasses import dataclass
import numpy as np
from .frames import Frame

@dataclass(frozen=True)
class WindowSpec:
    width: float     # m  # 中文：窗宽（米）
    height: float    # m  # 中文：窗高（米）
    sill_h: float    # m above floor (bottom of window)  # 中文：窗台高度（窗底距地/楼面高度，米）
    floor_z: float   # world z of the floor of this window's level  # 中文：该窗所在楼层楼面在世界坐标系中的 z 值

def rect_grid_samples_local(win: WindowSpec, nx: int, ny: int) -> np.ndarray:
    """
    Sample points on window plane (local frame): n=0
    local u in [0,width], local v in [sill_h, sill_h+height]
    Returns (N,3) local coords.

    中文：
    在窗面（局部坐标系，n=0 平面）上按矩形网格采样点：
      u ∈ [0, width]，v ∈ [sill_h, sill_h+height]。
    返回 (N,3) 的局部坐标点集。
    """
    us = (np.arange(nx) + 0.5) / nx * win.width
    vs = win.sill_h + (np.arange(ny) + 0.5) / ny * win.height
    U, V = np.meshgrid(us, vs, indexing="xy")
    pts = np.stack([U.ravel(), V.ravel(), np.zeros(U.size)], axis=1)
    return pts

def samples_world(frame: Frame, local_pts: np.ndarray) -> np.ndarray:
    return np.vstack([frame.local_to_world_pt(p) for p in local_pts])
