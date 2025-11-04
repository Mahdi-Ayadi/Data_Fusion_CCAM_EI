from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .math_utils import (
    MIN_VARIANCE,
    angle_diff,
    quat_to_yaw,
    sigma_from_uncertainties,
    weighted_circular_mean,
    yaw_to_quat,
)
from .models import Object3d


class KalmanTrack:
    """Simple constant-velocity Kalman filter over (x,y,z) + box sizes + yaw."""

    STATE_DIM = 10  # x, y, z, vx, vy, vz, w, l, h, yaw
    MEAS_DIM = 7    # x, y, z, w, l, h, yaw

    def __init__(
        self,
        seed_obj: Object3d,
        process_var_pos: float = 1.0,
        process_var_vel: float = 3.0,
        process_var_size: float = 0.5,
        process_var_yaw: float = np.deg2rad(5.0),
    ):
        self.instance_token = seed_obj.instance_token
        self.classification = seed_obj.classification
        self.last_timestamp = self._timestamp_to_seconds(seed_obj.timestamp)
        self.object_id = seed_obj.object_id
        self.state = np.zeros(self.STATE_DIM, dtype=float)
        self.P = np.eye(self.STATE_DIM, dtype=float)

        yaw = quat_to_yaw(seed_obj.orientation)
        self.state[:] = [
            seed_obj.xdistance,
            seed_obj.ydistance,
            seed_obj.zdistance,
            0.0,
            0.0,
            0.0,
            seed_obj.width,
            seed_obj.depth,
            seed_obj.height,
            yaw,
        ]

        sigmas = [
            sigma_from_uncertainties(seed_obj, 0, default=0.5),
            sigma_from_uncertainties(seed_obj, 1, default=0.5),
            sigma_from_uncertainties(seed_obj, 2, default=0.5),
            sigma_from_uncertainties(seed_obj, 4, default=0.5),
            sigma_from_uncertainties(seed_obj, 5, default=0.5),
            sigma_from_uncertainties(seed_obj, 6, default=0.5),
            sigma_from_uncertainties(seed_obj, 3, default=np.deg2rad(5.0)),
        ]
        self.P = np.diag(
            [
                sigmas[0] ** 2,
                sigmas[1] ** 2,
                sigmas[2] ** 2,
                process_var_vel,
                process_var_vel,
                process_var_vel,
                sigmas[3] ** 2,
                sigmas[4] ** 2,
                sigmas[5] ** 2,
                sigmas[6] ** 2,
            ]
        )

        self.process_var_pos = process_var_pos
        self.process_var_vel = process_var_vel
        self.process_var_size = process_var_size
        self.process_var_yaw = process_var_yaw

    @staticmethod
    def _timestamp_to_seconds(ts: float) -> float:
        if ts is None or np.isnan(ts):
            return np.nan
        ts = float(ts)
        if abs(ts) > 1e6:
            return ts * 1e-6
        return ts

    def _build_transition(self, dt: float) -> np.ndarray:
        F = np.eye(self.STATE_DIM, dtype=float)
        for idx in range(3):
            F[idx, idx + 3] = dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        q = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=float)
        pos_noise = self.process_var_pos * dt * dt
        vel_noise = self.process_var_vel * dt
        size_noise = self.process_var_size * dt
        q[0, 0] = q[1, 1] = q[2, 2] = pos_noise
        q[3, 3] = q[4, 4] = q[5, 5] = vel_noise
        q[6, 6] = q[7, 7] = q[8, 8] = size_noise
        q[9, 9] = (self.process_var_yaw * dt) ** 2
        return q

    def predict(self, timestamp: float, default_dt: float = 0.5):
        ts_sec = self._timestamp_to_seconds(timestamp)
        if np.isnan(self.last_timestamp) or np.isnan(ts_sec):
            dt = default_dt
        else:
            dt = max(ts_sec - self.last_timestamp, 1e-3)
        F = self._build_transition(dt)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self._process_noise(dt)
        self.last_timestamp = ts_sec if not np.isnan(ts_sec) else self.last_timestamp

    def _measurement_matrix(self) -> np.ndarray:
        H = np.zeros((self.MEAS_DIM, self.STATE_DIM), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 6] = 1.0
        H[4, 7] = 1.0
        H[5, 8] = 1.0
        H[6, 9] = 1.0
        return H

    def _measurement_vector(self, obj: Object3d) -> np.ndarray:
        return np.array(
            [
                obj.xdistance,
                obj.ydistance,
                obj.zdistance,
                obj.width,
                obj.depth,
                obj.height,
                quat_to_yaw(obj.orientation),
            ],
            dtype=float,
        )

    def _measurement_covariance(self, obj: Object3d) -> np.ndarray:
        return np.diag(
            [
                sigma_from_uncertainties(obj, 0, default=0.5) ** 2,
                sigma_from_uncertainties(obj, 1, default=0.5) ** 2,
                sigma_from_uncertainties(obj, 2, default=0.5) ** 2,
                sigma_from_uncertainties(obj, 4, default=0.5) ** 2,
                sigma_from_uncertainties(obj, 5, default=0.5) ** 2,
                sigma_from_uncertainties(obj, 6, default=0.5) ** 2,
                sigma_from_uncertainties(obj, 3, default=np.deg2rad(5.0)) ** 2,
            ]
        )

    def update(self, obj: Object3d):
        H = self._measurement_matrix()
        z = self._measurement_vector(obj)
        R = self._measurement_covariance(obj)
        y = z - (H @ self.state)
        y[-1] = angle_diff(z[-1], (H @ self.state)[-1])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(self.STATE_DIM)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        self.classification = obj.classification or self.classification
        if obj.object_id:
            self.object_id = obj.object_id

    def to_object3d(self, sample_token: str, timestamp: float) -> Object3d:
        yaw = self.state[9]
        orientation = yaw_to_quat(yaw)
        uncertainties = [
            float(np.sqrt(max(self.P[0, 0], MIN_VARIANCE))),
            float(np.sqrt(max(self.P[1, 1], MIN_VARIANCE))),
            float(np.sqrt(max(self.P[2, 2], MIN_VARIANCE))),
            float(np.sqrt(max(self.P[9, 9], MIN_VARIANCE))),
            float(np.sqrt(max(self.P[6, 6], MIN_VARIANCE))),
            float(np.sqrt(max(self.P[7, 7], MIN_VARIANCE))),
            float(np.sqrt(max(self.P[8, 8], MIN_VARIANCE))),
        ]
        return Object3d(
            id_frame=sample_token,
            instance_token=self.instance_token,
            object_id=self.object_id,
            timestamp=timestamp,
            xdistance=float(self.state[0]),
            ydistance=float(self.state[1]),
            zdistance=float(self.state[2]),
            xspeed=float(self.state[3]),
            yspeed=float(self.state[4]),
            zspeed=float(self.state[5]),
            width=float(self.state[6]),
            depth=float(self.state[7]),
            height=float(self.state[8]),
            orientation=orientation,
            classification=self.classification,
            uncertainties=uncertainties,
            dt=0.5,
        )


def run_kalman_fusion(
    sample_tokens: Iterable[str],
    noisy_low_by_sample: Dict[str, List[Object3d]],
    noisy_high_by_sample: Dict[str, List[Object3d]],
    matches_by_sample: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, List[Object3d]]:
    """Apply Kalman fusion track-wise using matched detections between sensors."""
    tracks: Dict[str, KalmanTrack] = {}
    fused_output: Dict[str, List[Object3d]] = defaultdict(list)

    for sample_token in sample_tokens:
        noisy_low = noisy_low_by_sample.get(sample_token, [])
        noisy_high = noisy_high_by_sample.get(sample_token, [])
        matches = matches_by_sample.get(sample_token, [])

        timestamp = np.nan
        if noisy_low:
            timestamp = noisy_low[0].timestamp
        elif noisy_high:
            timestamp = noisy_high[0].timestamp

        for idx_low, idx_high in matches:
            meas_low = noisy_low[idx_low]
            meas_high = noisy_high[idx_high]
            track_id = meas_low.instance_token or meas_low.object_id

            if track_id not in tracks:
                tracks[track_id] = KalmanTrack(meas_low)

            track = tracks[track_id]
            track.predict(timestamp=timestamp if not np.isnan(timestamp) else meas_low.timestamp)
            track.update(meas_low)
            track.update(meas_high)
            fused_output[sample_token].append(track.to_object3d(sample_token, meas_low.timestamp))

    return dict(fused_output)
