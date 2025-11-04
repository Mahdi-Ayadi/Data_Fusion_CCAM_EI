from __future__ import annotations

import json
import os
from collections import defaultdict
from math import pi
from typing import Dict, List, Tuple

import numpy as np

from .math_utils import quat_to_yaw, yaw_to_quat
from .models import Object3d, Quaternion


def load_nuscenes_gt(
    nuscenes_ann_root: str,
    prefer_channel: str = "LIDAR_TOP",
    fraction: float = 1.0,
) -> Dict[str, List[Object3d]]:
    """Load nuScenes ground truth annotations."""

    def _load(name: str):
        with open(os.path.join(nuscenes_ann_root, f"{name}.json"), "r") as f:
            return json.load(f)

    samples = _load("sample")
    sample_annotations = _load("sample_annotation")
    sample_data = _load("sample_data")

    n_keep = int(len(samples) * fraction)
    samples = samples[:n_keep]
    sample_tokens_kept = {s["token"] for s in samples}

    anns_by_sample: Dict[str, List[dict]] = defaultdict(list)
    for ann in sample_annotations:
        if ann["sample_token"] in sample_tokens_kept:
            anns_by_sample[ann["sample_token"]].append(ann)

    sd_by_sample: Dict[str, List[dict]] = defaultdict(list)
    for sd in sample_data:
        if sd["sample_token"] in sample_tokens_kept:
            sd_by_sample[sd["sample_token"]].append(sd)

    sample_ts: Dict[str, float] = {}
    for sample_token, sds in sd_by_sample.items():
        chosen = None
        for sd in sds:
            if sd.get("channel") == prefer_channel:
                chosen = sd
                break
        if chosen is None and sds:
            chosen = sds[0]
        sample_ts[sample_token] = chosen["timestamp"] if chosen else np.nan

    objects_by_sample: Dict[str, List[Object3d]] = defaultdict(list)
    for sample_token, anns in anns_by_sample.items():
        ts = sample_ts.get(sample_token, np.nan)
        for ann in anns:
            x, y, z = ann["translation"]
            w, l, h = ann["size"]
            qw, qx, qy, qz = ann["rotation"]
            q = Quaternion(qw, qx, qy, qz)
            cat = ann.get("category_name")
            obj = Object3d(
                id_frame=sample_token,
                instance_token=ann.get("instance_token"),
                object_id=ann["token"],
                timestamp=ts,
                xdistance=float(x),
                ydistance=float(y),
                zdistance=float(z),
                width=float(w),
                depth=float(l),
                height=float(h),
                orientation=q,
                classification=cat,
                uncertainties=None,
            )
            objects_by_sample[sample_token].append(obj)
    return objects_by_sample


def add_gaussian_noise(
    objs: List[Object3d],
    sigmas: dict,
    rnd: np.random.RandomState = None,
) -> List[Object3d]:
    """Return new list with Gaussian noise applied; populate uncertainties."""
    rnd = rnd or np.random.RandomState(0)

    out: List[Object3d] = []
    for o in objs:
        yaw = quat_to_yaw(o.orientation)

        nx = rnd.normal(scale=sigmas.get("x", 0.0))
        ny = rnd.normal(scale=sigmas.get("y", 0.0))
        nz = rnd.normal(scale=sigmas.get("z", 0.0))
        nyaw = rnd.normal(scale=sigmas.get("yaw", 0.0))
        nw = rnd.normal(scale=sigmas.get("w", 0.0))
        nl = rnd.normal(scale=sigmas.get("l", 0.0))
        nh = rnd.normal(scale=sigmas.get("h", 0.0))

        new_w = max(1e-3, o.width + nw)
        new_l = max(1e-3, o.depth + nl)
        new_h = max(1e-3, o.height + nh)

        new_yaw = yaw + nyaw
        if new_yaw > pi:
            new_yaw -= 2 * pi
        if new_yaw <= -pi:
            new_yaw += 2 * pi

        new_q = yaw_to_quat(new_yaw)

        o2 = Object3d(
            id_frame=o.id_fame,
            instance_token=o.instance_token,
            object_id=o.object_id,
            timestamp=o.timestamp,
            xdistance=o.xdistance + nx,
            ydistance=o.ydistance + ny,
            zdistance=o.zdistance + nz,
            xspeed=o.xspeed,
            yspeed=o.yspeed,
            zspeed=o.zspeed,
            width=new_w,
            depth=new_l,
            height=new_h,
            classification=o.classification,
            orientation=new_q,
            uncertainties=[
                sigmas.get("x", 0.0),
                sigmas.get("y", 0.0),
                sigmas.get("z", 0.0),
                sigmas.get("yaw", 0.0),
                sigmas.get("w", 0.0),
                sigmas.get("l", 0.0),
                sigmas.get("h", 0.0),
            ],
            dt=o.dt,
        )
        out.append(o2)
    return out


def make_two_noise_levels(
    objects_by_sample: Dict[str, List[Object3d]],
    sigmas_low: dict,
    sigmas_high: dict,
    seed: int = 0,
) -> Dict[str, Tuple[List[Object3d], List[Object3d]]]:
    """Create paired noisy detections for two sensors."""
    rnd_low = np.random.RandomState(seed)
    rnd_high = np.random.RandomState(seed + 1)
    per_sample: Dict[str, Tuple[List[Object3d], List[Object3d]]] = {}
    for sample_token, objs in objects_by_sample.items():
        noisy_low = add_gaussian_noise(objs, sigmas_low, rnd_low)
        noisy_high = add_gaussian_noise(objs, sigmas_high, rnd_high)
        per_sample[sample_token] = (noisy_low, noisy_high)
    return per_sample
