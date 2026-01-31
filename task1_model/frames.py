# task1_model/frames.py
from dataclasses import dataclass
import numpy as np

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

@dataclass(frozen=True)
class Frame:
    """
    Window-local frame:
      origin: 3D point
      u: window width axis (horizontal along facade)
      v: vertical up
      n: outward normal (points outside)
    """
    origin: np.ndarray  # (3,)
    u: np.ndarray       # (3,)
    v: np.ndarray       # (3,)
    n: np.ndarray       # (3,)

    @staticmethod
    def from_facade_azimuth(origin: np.ndarray, facade_azimuth_deg: float) -> "Frame":
        """
        ENU coordinate convention:
          x = East, y = North, z = Up
        facade_azimuth_deg is outward normal azimuth:
          0=N, 90=E, 180=S, 270=W
        """
        az = np.deg2rad(facade_azimuth_deg)
        n = _unit(np.array([np.sin(az), np.cos(az), 0.0]))  # outward in horizontal plane
        v = np.array([0.0, 0.0, 1.0])                       # up
        # u is to the right when facing outward: u = v x n
        u = _unit(np.cross(v, n))
        return Frame(origin=np.asarray(origin, float), u=u, v=v, n=n)

    def world_to_local_vec(self, vec: np.ndarray) -> np.ndarray:
        """Transform a direction vector (world) to local components [u,v,n]."""
        vec = np.asarray(vec, float)
        return np.array([np.dot(vec, self.u), np.dot(vec, self.v), np.dot(vec, self.n)])

    def world_to_local_pt(self, pt: np.ndarray) -> np.ndarray:
        """Transform a point (world) to local coordinates."""
        pt = np.asarray(pt, float) - self.origin
        return np.array([np.dot(pt, self.u), np.dot(pt, self.v), np.dot(pt, self.n)])

    def local_to_world_pt(self, local_pt: np.ndarray) -> np.ndarray:
        """Local point [u,v,n] to world."""
        x, y, z = local_pt
        return self.origin + x*self.u + y*self.v + z*self.n

    def local_to_world_vec(self, local_vec: np.ndarray) -> np.ndarray:
        """Local direction [u,v,n] to world."""
        x, y, z = local_vec
        return x*self.u + y*self.v + z*self.n
