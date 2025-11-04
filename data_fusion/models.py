from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np

try:
    from pyquaternion import Quaternion as _Quaternion  # type: ignore
except ModuleNotFoundError:
    _Quaternion = None


class Quaternion:
    """Minimal quaternion helper covering the needs of this project."""

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self._normalize()

    def _normalize(self) -> None:
        n = sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if n == 0.0:
            self.w, self.x, self.y, self.z = 1.0, 0.0, 0.0, 0.0
            return
        inv = 1.0 / n
        self.w *= inv
        self.x *= inv
        self.y *= inv
        self.z *= inv

    @property
    def rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(1.0, 0.0, 0.0, 0.0)

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.3f}, x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


if _Quaternion is not None:
    Quaternion = _Quaternion  # type: ignore


class Object3d:
    """Container for 3D object detections with optional uncertainty metadata."""

    def __init__(
        self,
        id_frame: Optional[str] = None,
        instance_token: Optional[str] = None,
        uncertainties: Optional[list] = None,
        orientation: Optional[Quaternion] = None,
        object_id: Optional[str] = None,
        timestamp: float = np.nan,
        xdistance: float = np.nan,
        ydistance: float = np.nan,
        zdistance: float = np.nan,
        xspeed: float = np.nan,
        yspeed: float = np.nan,
        zspeed: float = np.nan,
        width: float = np.nan,
        height: float = np.nan,
        depth: float = np.nan,
        classification: Optional[str] = None,
        distance_range: Optional[float] = None,
        dt: float = 0.5,
    ):
        self.object_id = object_id
        self.object_distance_range = distance_range
        self.id_fame = id_frame  # keeping legacy attribute name
        self.instance_token = instance_token
        self.timestamp = timestamp

        self.xdistance = xdistance
        self.ydistance = ydistance
        self.zdistance = zdistance
        self.xspeed = xspeed
        self.yspeed = yspeed
        self.zspeed = zspeed
        self.orientation = orientation or Quaternion(1, 0, 0, 0)

        self.width = width
        self.height = height
        self.depth = depth
        self.classification = classification
        self.uncertainties = uncertainties
        self.feature_frame = None
        self.dt = dt

    def __repr__(self) -> str:
        return (
            f"Object3d(id_frame={self.id_fame}, object_id={self.object_id}, "
            f"class={self.classification}, pos=({self.xdistance:.2f}, "
            f"{self.ydistance:.2f}, {self.zdistance:.2f}))"
        )
