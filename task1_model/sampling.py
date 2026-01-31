# task1_model/sampling.py
from dataclasses import dataclass
import numpy as np
from .frames import Frame

@dataclass(frozen=True)
class WindowSpec:
    width: float     # m
    height: float    # m
    sill_h: float    # m above floor (bottom of window)
    floor_z: float   # world z of the floor of this window's level

def rect_grid_samples_local(win: WindowSpec, nx: int, ny: int) -> np.ndarray:
    """
    Sample points on window plane (local frame): n=0
    local u in [0,width], local v in [sill_h, sill_h+height]
    Returns (N,3) local coords.
    """
    us = (np.arange(nx) + 0.5) / nx * win.width
    vs = win.sill_h + (np.arange(ny) + 0.5) / ny * win.height
    U, V = np.meshgrid(us, vs, indexing="xy")
    pts = np.stack([U.ravel(), V.ravel(), np.zeros(U.size)], axis=1)
    return pts

def samples_world(frame: Frame, local_pts: np.ndarray) -> np.ndarray:
    return np.vstack([frame.local_to_world_pt(p) for p in local_pts])
