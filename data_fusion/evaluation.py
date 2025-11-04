from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .math_utils import angle_diff, quat_to_yaw
from .models import Object3d


def evaluate_against_gt(fused_by_sample: Dict[str, List[Object3d]], gt_by_sample: Dict[str, List[Object3d]]) -> dict:
    """Compute simple quality indicators for fused detections against ground truth boxes."""
    translation_l2 = []
    translation_sq = []
    translation_l1 = []
    yaw_deg_abs = []
    size_l1 = []
    matched = 0

    for sample_token, fused_objs in fused_by_sample.items():
        gt_lookup = {obj.object_id: obj for obj in gt_by_sample.get(sample_token, [])}
        for fused in fused_objs:
            if not fused.object_id:
                continue
            parts = str(fused.object_id).split("|")
            gt_obj = None
            for part in parts:
                if part in gt_lookup:
                    gt_obj = gt_lookup[part]
                    break
            if gt_obj is None:
                continue

            matched += 1
            dx = fused.xdistance - gt_obj.xdistance
            dy = fused.ydistance - gt_obj.ydistance
            dz = fused.zdistance - gt_obj.zdistance

            l2 = hypot(hypot(dx, dy), dz)
            translation_l2.append(l2)
            translation_sq.append(l2 * l2)
            translation_l1.append((abs(dx) + abs(dy) + abs(dz)) / 3.0)

            yaw_fused = quat_to_yaw(fused.orientation)
            yaw_gt = quat_to_yaw(gt_obj.orientation)
            yaw_deg_abs.append(abs(np.rad2deg(angle_diff(yaw_fused, yaw_gt))))

            dw = abs(fused.width - gt_obj.width)
            dl = abs(fused.depth - gt_obj.depth)
            dh = abs(fused.height - gt_obj.height)
            size_l1.append((dw + dl + dh) / 3.0)

    if matched == 0:
        return {
            "matched": 0,
            "translation_l2_mean": float("nan"),
            "translation_l2_rmse": float("nan"),
            "translation_l1_mean": float("nan"),
            "yaw_deg_mae": float("nan"),
            "size_l1_mean": float("nan"),
        }

    return {
        "matched": matched,
        "translation_l2_mean": float(np.mean(translation_l2)),
        "translation_l2_rmse": float(np.sqrt(np.mean(translation_sq))),
        "translation_l1_mean": float(np.mean(translation_l1)),
        "yaw_deg_mae": float(np.mean(yaw_deg_abs)),
        "size_l1_mean": float(np.mean(size_l1)),
    }


def print_metrics(label: str, metrics: dict) -> None:
    """Nicely format metric dictionaries."""
    print(f"Metrics - {label}")
    print(f"  Matched boxes      : {metrics['matched']}")
    print(f"  Position L2 mean   : {metrics['translation_l2_mean']:.3f} m")
    print(f"  Position L2 RMSE   : {metrics['translation_l2_rmse']:.3f} m")
    print(f"  Position L1 mean   : {metrics['translation_l1_mean']:.3f} m")
    print(f"  Yaw MAE            : {metrics['yaw_deg_mae']:.3f} °")
    print(f"  Size L1 mean       : {metrics['size_l1_mean']:.3f} m")
    print("")


def print_nuscenes_metrics(label: str, metrics: dict) -> None:
    """Pretty-print a subset of nuScenes-style metrics."""
    print(f"nuScenes-style metrics - {label}")
    print(f"  mAP   : {metrics['mAP']:.3f}")
    print(f"  mATE  : {metrics['mATE']:.3f} m")
    print(f"  mASE  : {metrics['mASE']:.3f}")
    print(f"  mAOE  : {metrics['mAOE']:.3f} rad")
    if not np.isnan(metrics.get("mAVE", float("nan"))):
        print(f"  mAVE  : {metrics['mAVE']:.3f} m/s")
    print("")


def associate_to_gt_per_frame_nuscenes_like(
    dets: List[Object3d],
    gts: List[Object3d],
    dist_thresh: float = 2.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    One-to-one greedy matching by 2D center distance, per class.
    Returns matches and unmatched indices.
    """
    idx_det_by_cls = defaultdict(list)
    idx_gt_by_cls = defaultdict(list)
    for i, d in enumerate(dets):
        idx_det_by_cls[d.classification].append(i)
    for j, g in enumerate(gts):
        idx_gt_by_cls[g.classification].append(j)

    matches = []
    used_det, used_gt = set(), set()

    for cls in set(list(idx_det_by_cls.keys()) + list(idx_gt_by_cls.keys())):
        det_idx = idx_det_by_cls.get(cls, [])
        gt_idx = idx_gt_by_cls.get(cls, [])
        if not det_idx or not gt_idx:
            continue

        pairs = []
        for i in det_idx:
            for j in gt_idx:
                d = hypot(dets[i].xdistance - gts[j].xdistance, dets[i].ydistance - gts[j].ydistance)
                if d <= dist_thresh:
                    pairs.append((d, i, j))
        pairs.sort(key=lambda x: x[0])

        for d, i, j in pairs:
            if i in used_det or j in used_gt:
                continue
            used_det.add(i)
            used_gt.add(j)
            matches.append((i, j))

    unmatched_det = [i for i in range(len(dets)) if i not in used_det]
    unmatched_gt = [j for j in range(len(gts)) if j not in used_gt]
    return matches, unmatched_det, unmatched_gt


def iou2d_sizes_aligned(det_w: float, det_l: float, gt_w: float, gt_l: float) -> float:
    """IoU of rectangles when centers & yaw are aligned (nuScenes ASE proxy)."""
    inter = min(det_w, gt_w) * min(det_l, gt_l)
    area_d = det_w * det_l
    area_g = gt_w * gt_l
    union = area_d + area_g - inter
    return inter / union if union > 0 else 0.0


def pr_ap(points: Iterable[Tuple[float, float]]) -> float:
    """Compute AP from (precision, recall) points using 101-pt interpolation."""
    points = list(points)
    if not points:
        return 0.0
    recalls = np.array([r for _, r in points])
    precisions = np.array([p for p, _ in points])
    mpre = np.maximum.accumulate(precisions[::-1])[::-1]
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = mpre[recalls >= t].max() if np.any(recalls >= t) else 0.0
        ap += p / 101.0
    return float(ap)


def evaluate_nuscenes_like(
    gt_by_sample: Dict[str, List[Object3d]],
    det_by_sample: Dict[str, List[Object3d]],
    scores_by_sample: Dict[str, List[float]] = None,
    dist_thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> dict:
    """Compute nuScenes-style metrics (mAP, mATE, mASE, mAOE, mAVE)."""
    classes = sorted({o.classification for objs in gt_by_sample.values() for o in objs})

    ap_data = {cls: {thr: {"pairs": [], "num_gt": 0} for thr in dist_thresholds} for cls in classes}

    THR_TP = 2.0
    tp_errs = {cls: {"ATE": [], "ASE": [], "AOE": [], "AVE": []} for cls in classes}

    gt_count_by_cls = {cls: 0 for cls in classes}
    for objs in gt_by_sample.values():
        for g in objs:
            gt_count_by_cls[g.classification] += 1
    for cls in classes:
        for thr in dist_thresholds:
            ap_data[cls][thr]["num_gt"] = gt_count_by_cls[cls]

    for token, gts in gt_by_sample.items():
        dets = det_by_sample.get(token, [])
        scores = scores_by_sample.get(token) if scores_by_sample else None
        if scores is None:
            scores = [1.0] * len(dets)

        order = np.argsort(-np.array(scores))
        dets_sorted = [dets[i] for i in order]
        scores_sorted = [scores[i] for i in order]

        assoc_per_thr = {}
        for thr in dist_thresholds:
            matches, _, _ = associate_to_gt_per_frame_nuscenes_like(dets_sorted, gts, dist_thresh=thr)
            assoc_per_thr[thr] = matches

        for thr in dist_thresholds:
            det_to_gt = {i_d: i_g for (i_d, i_g) in assoc_per_thr[thr]}
            for i_d, det in enumerate(dets_sorted):
                cls = det.classification
                is_tp = 1 if i_d in det_to_gt else 0
                ap_data[cls][thr]["pairs"].append((scores_sorted[i_d], is_tp))

        matches_2m, _, _ = associate_to_gt_per_frame_nuscenes_like(dets_sorted, gts, dist_thresh=THR_TP)
        for i_d, i_g in matches_2m:
            d = dets_sorted[i_d]
            g = gts[i_g]
            ate = hypot(d.xdistance - g.xdistance, d.ydistance - g.ydistance)
            iou = iou2d_sizes_aligned(d.width, d.depth, g.width, g.depth)
            ase = 1.0 - iou
            aoe = abs(angle_diff(quat_to_yaw(d.orientation), quat_to_yaw(g.orientation)))
            if not np.isnan(d.xspeed) and not np.isnan(g.xspeed):
                vx_err = d.xspeed - g.xspeed
                vy_err = d.yspeed - g.yspeed
                ave = np.hypot(vx_err, vy_err)
            else:
                ave = None

            cls = g.classification
            tp_errs[cls]["ATE"].append(ate)
            tp_errs[cls]["ASE"].append(ase)
            tp_errs[cls]["AOE"].append(aoe)
            if ave is not None:
                tp_errs[cls]["AVE"].append(ave)

    ap_per_cls = {cls: [] for cls in classes}
    for cls in classes:
        for thr in dist_thresholds:
            pairs = ap_data[cls][thr]["pairs"]
            num_gt = ap_data[cls][thr]["num_gt"]
            if num_gt == 0:
                ap_per_cls[cls].append(0.0)
                continue
            pairs.sort(key=lambda x: -x[0])
            tps = np.cumsum([tp for _, tp in pairs])
            fps = np.cumsum([1 - tp for _, tp in pairs])
            recalls = tps / max(num_gt, 1)
            precisions = tps / np.maximum(tps + fps, 1e-9)
            pr_points = list(zip(precisions, recalls))
            pr_points.sort(key=lambda pr: pr[1])
            ap = pr_ap(pr_points)
            ap_per_cls[cls].append(ap)

    def _mean(xs):
        arr = [x for x in xs if not np.isnan(x)]
        return float(np.mean(arr)) if arr else float("nan")

    mAP_per_cls = {cls: float(np.mean(ap_per_cls[cls])) if ap_per_cls[cls] else 0.0 for cls in classes}
    mAP = float(np.mean(list(mAP_per_cls.values()))) if classes else 0.0

    mATE_per_cls = {cls: _mean(tp_errs[cls]["ATE"]) for cls in classes}
    mASE_per_cls = {cls: _mean(tp_errs[cls]["ASE"]) for cls in classes}
    mAOE_per_cls = {cls: _mean(tp_errs[cls]["AOE"]) for cls in classes}
    mAVE_per_cls = {cls: _mean(tp_errs[cls]["AVE"]) for cls in classes if tp_errs[cls]["AVE"]}

    results = {
        "mAP": mAP,
        "mAP_per_class": mAP_per_cls,
        "mATE": _mean(mATE_per_cls.values()),
        "mASE": _mean(mASE_per_cls.values()),
        "mAOE": _mean(mAOE_per_cls.values()),
        "mAVE": _mean(mAVE_per_cls.values()) if mAVE_per_cls else float("nan"),
        "per_class": {
            "ATE": mATE_per_cls,
            "ASE": mASE_per_cls,
            "AOE": mAOE_per_cls,
            "AVE": mAVE_per_cls,
        },
        "details": {
            "dist_thresholds": dist_thresholds,
            "tp_errors_raw": tp_errs,
        },
    }
    return results
