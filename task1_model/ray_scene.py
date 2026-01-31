# task1_model/ray_scene.py
from dataclasses import dataclass
import numpy as np
import trimesh

@dataclass
class RayScene:
    mesh_local: trimesh.Trimesh

    @staticmethod
    def from_meshes(meshes: list[trimesh.Trimesh]) -> "RayScene":
        meshes = [m for m in meshes if m is not None and len(m.vertices) > 0]
        if not meshes:
            empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)
            return RayScene(empty)
        return RayScene(trimesh.util.concatenate(meshes))

    def shaded_mask(self, origins_local: np.ndarray, sun_dir_local: np.ndarray) -> np.ndarray:
        """
        Return boolean mask: True if point is shaded (ray hits mesh), False if lit.
        sun_dir_local points from window point toward the sun (outward).
        """
        if len(self.mesh_local.vertices) == 0:
            return np.zeros((origins_local.shape[0],), dtype=bool)

        # trimesh expects (N,3) origins and (N,3) directions
        dirs = np.repeat(sun_dir_local.reshape(1, 3), origins_local.shape[0], axis=0)

        # intersects_any -> True if hit any triangle
        try:
            hit = self.mesh_local.ray.intersects_any(origins_local, dirs)
        except BaseException:
            # Fallback: slower but robust
            locations, index_ray, _ = self.mesh_local.ray.intersects_location(
                origins_local, dirs, multiple_hits=False
            )
            hit = np.zeros((origins_local.shape[0],), dtype=bool)
            hit[index_ray] = True

        return hit
