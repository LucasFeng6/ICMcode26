# 坐标系/窗户局部坐标框架定义
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

    窗户局部坐标系：
      origin：原点（3D 点）
      u：窗宽方向（沿立面水平）
      v：竖直向上
      n：外法线方向（指向室外）
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

        ENU 坐标约定：x=东，y=北，z=上。
        facade_azimuth_deg 表示立面外法线的方位角：0=北，90=东，180=南，270=西。
        """
        az = np.deg2rad(facade_azimuth_deg)
        n = _unit(np.array([np.sin(az), np.cos(az), 0.0]))  # 水平面内的外法线方向
        v = np.array([0.0, 0.0, 1.0])                       # 竖直向上
        # 面向室外时，u 指向右侧：u = v × n
        u = _unit(np.cross(v, n))
        return Frame(origin=np.asarray(origin, float), u=u, v=v, n=n)

    def world_to_local_vec(self, vec: np.ndarray) -> np.ndarray:
        """Transform a direction vector (world) to local components [u,v,n].  中文：将世界坐标系方向向量转换为局部坐标分量 [u,v,n]"""
        vec = np.asarray(vec, float)
        return np.array([np.dot(vec, self.u), np.dot(vec, self.v), np.dot(vec, self.n)])

    def world_to_local_pt(self, pt: np.ndarray) -> np.ndarray:
        """Transform a point (world) to local coordinates.  中文：将世界坐标系中的点转换到局部坐标系"""
        pt = np.asarray(pt, float) - self.origin
        return np.array([np.dot(pt, self.u), np.dot(pt, self.v), np.dot(pt, self.n)])

    def local_to_world_pt(self, local_pt: np.ndarray) -> np.ndarray:
        """Local point [u,v,n] to world.  中文：将局部坐标点 [u,v,n] 转换为世界坐标点"""
        x, y, z = local_pt
        return self.origin + x*self.u + y*self.v + z*self.n

    def local_to_world_vec(self, local_vec: np.ndarray) -> np.ndarray:
        """Local direction [u,v,n] to world.  中文：将局部坐标方向 [u,v,n] 转换为世界坐标方向"""
        x, y, z = local_vec
        return x*self.u + y*self.v + z*self.n
