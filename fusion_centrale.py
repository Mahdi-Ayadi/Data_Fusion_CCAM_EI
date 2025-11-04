import json
import os
from collections import defaultdict
import numpy as np
from pyquaternion import Quaternion
from math import atan2, sin, cos, pi, hypot


# ---- Your class with minimal safe tweaks (kept shape & names) ----
class Object3d:
    def __init__(
        self,
        id_frame: str = None,
        uncertainties: list = None,
        orientation: Quaternion = None,
        object_id: str = None,
        timestamp: int = np.nan,
        xdistance: float = np.nan,
        ydistance: float = np.nan,
        zdistance: float = np.nan,
        xspeed: float = np.nan,
        yspeed: float = np.nan,
        zspeed: float = np.nan,
        width: float = np.nan,
        height: float = np.nan,
        depth: float = np.nan,
        classification: str = None,
        distance_range: float = None,
        dt=0.5,
    ):
        self.object_id = object_id
        self.object_distance_range = distance_range
        self.id_fame = id_frame  # (kept your original attribute name)
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
        self.feature_frame = None  # was undefined; set to None explicitly
        self.dt = dt


# ---------- Utilities ----------
def quat_to_yaw(q: Quaternion) -> float:
    """
    Extract yaw (heading around Z) from a quaternion, in radians, wrapped to [-pi, pi].
    Assumes nuScenes ENU-like convention (z-up). Adjust if your frame differs.
    """
    # yaw from quaternion (intrinsic ZYX), here we only need Z rotation component
    # q.rotation_matrix -> extract yaw via atan2
    R = q.rotation_matrix
    yaw = atan2(R[1,0], R[0,0])
    # wrap to [-pi, pi]
    if yaw > pi:
        yaw -= 2*pi
    if yaw <= -pi:
        yaw += 2*pi
    return yaw

def yaw_to_quat(yaw: float) -> Quaternion:
    """Build a quaternion representing a pure yaw about Z axis."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    # yaw about Z: q = [w, x, y, z] = [cos(y/2), 0, 0, sin(y/2)]
    return Quaternion(cy, 0.0, 0.0, sy)

def circular_mean(yaws: np.ndarray) -> float:
    """Mean of angles in radians, robust to wrap-around."""
    s = np.mean(np.sin(yaws))
    c = np.mean(np.cos(yaws))
    return atan2(s, c)

# ---------- 1) Load nuScenes GT into Object3d ----------
def load_nuscenes_gt(nuscenes_ann_root: str,
                     prefer_channel: str = "LIDAR_TOP",
                     fraction: float = 1.0) -> dict:
    """
    Reads nuScenes JSONs (v1.0-* style) from a folder and returns:
    dict[sample_token] -> list[Object3d]

    Expected files in `nuscenes_ann_root`:
      - sample.json
      - sample_annotation.json
      - sample_data.json

    Notes:
    - For each sample, we pick a timestamp from the sample_data belonging to
      `prefer_channel` if available; otherwise the first sample_data of the sample.
    - size in nuScenes is [w, l, h]. We map: width=w, depth=l, height=h.
    - rotation is a quaternion [w, x, y, z] and fed directly to pyquaternion.
    """
    def _load(name):
        with open(os.path.join(nuscenes_ann_root, f"{name}.json"), "r") as f:
            return json.load(f)

    samples = _load("sample")
    sample_annotations = _load("sample_annotation")
    sample_data = _load("sample_data")

    # ---- subsample frames (samples) ----
    n_keep = int(len(samples) * fraction)
    samples = samples[:n_keep]  # or random.sample(samples, n_keep) for random subset
    sample_tokens_kept = {s["token"] for s in samples}

    anns_by_sample = defaultdict(list)
    for ann in sample_annotations:
        if ann["sample_token"] in sample_tokens_kept:
            anns_by_sample[ann["sample_token"]].append(ann)

    sd_by_sample = defaultdict(list)
    for sd in sample_data:
        if sd["sample_token"] in sample_tokens_kept:
            sd_by_sample[sd["sample_token"]].append(sd)

    # timestamps
    sample_ts = {}
    for sample_token, sds in sd_by_sample.items():
        chosen = None
        for sd in sds:
            if sd.get("channel") == prefer_channel:
                chosen = sd
                break
        if chosen is None and sds:
            chosen = sds[0]
        sample_ts[sample_token] = chosen["timestamp"] if chosen else np.nan

    from math import pi
    objects_by_sample = defaultdict(list)
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

    #print(f"Loaded {len(objects_by_sample)} samples (~{fraction*100:.0f}% of GT)")
    return objects_by_sample

# ---------- 2) Apply Gaussian noise & set uncertainties ----------
def add_gaussian_noise(objs: list,
                       sigmas: dict,
                       rnd: np.random.RandomState = None) -> list:
    """
    Returns a NEW list of Object3d with Gaussian noise applied.
    `sigmas` keys: 'x','y','z','yaw','w','l','h' (meters, radians for yaw).
    Also fills obj.uncertainties = [sigma_x, sigma_y, sigma_z, sigma_yaw, sigma_w, sigma_l, sigma_h].
    """
    rnd = rnd or np.random.RandomState(0)

    out = []
    for o in objs:
        yaw = quat_to_yaw(o.orientation)

        nx = rnd.normal(scale=sigmas.get('x', 0.0))
        ny = rnd.normal(scale=sigmas.get('y', 0.0))
        nz = rnd.normal(scale=sigmas.get('z', 0.0))
        nyaw = rnd.normal(scale=sigmas.get('yaw', 0.0))
        nw = rnd.normal(scale=sigmas.get('w', 0.0))
        nl = rnd.normal(scale=sigmas.get('l', 0.0))
        nh = rnd.normal(scale=sigmas.get('h', 0.0))

        # Ensure sizes remain positive
        new_w = max(1e-3, o.width + nw)
        new_l = max(1e-3, o.depth + nl)
        new_h = max(1e-3, o.height + nh)

        new_yaw = yaw + nyaw
        # wrap
        if new_yaw > pi:
            new_yaw -= 2*pi
        if new_yaw <= -pi:
            new_yaw += 2*pi

        new_q = yaw_to_quat(new_yaw)

        o2 = Object3d(
            id_frame=o.id_fame,
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
                sigmas.get('x', 0.0),
                sigmas.get('y', 0.0),
                sigmas.get('z', 0.0),
                sigmas.get('yaw', 0.0),
                sigmas.get('w', 0.0),
                sigmas.get('l', 0.0),
                sigmas.get('h', 0.0),
            ],
            dt=o.dt,
        )
        out.append(o2)
    return out

def make_two_noise_levels(objects_by_sample: dict,
                          sigmas_low: dict,
                          sigmas_high: dict,
                          seed: int = 0):
    """
    For each sample, return (noisy_low, noisy_high) lists.
    """
    rnd_low = np.random.RandomState(seed)
    rnd_high = np.random.RandomState(seed+1)
    per_sample = {}
    for sample_token, objs in objects_by_sample.items():
        noisy_low = add_gaussian_noise(objs, sigmas_low, rnd_low)
        noisy_high = add_gaussian_noise(objs, sigmas_high, rnd_high)
        per_sample[sample_token] = (noisy_low, noisy_high)
    return per_sample

# ---------- 3) Association (2D nearest neighbor) ----------
# TODO: TO BE REPLACED BY YOUR OWN ALGORITHM

def associate_greedy_2d(setA: list, setB: list, dist_thresh: float = 5.0):
    """
    Greedy nearest-neighbor association by (x,y) Euclidean distance.
    Returns:
      matches: list of (iA, iB)
      unmatched_A: indices
      unmatched_B: indices
    """
    A_xy = np.array([[o.xdistance, o.ydistance] for o in setA], dtype=float)
    B_xy = np.array([[o.xdistance, o.ydistance] for o in setB], dtype=float)

    usedA = set()
    usedB = set()
    matches = []

    # Build all pairs with distances
    pairs = []
    for i in range(len(setA)):
        for j in range(len(setB)):
            d = hypot(A_xy[i,0]-B_xy[j,0], A_xy[i,1]-B_xy[j,1])
            pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])

    for d, i, j in pairs:
        if d > dist_thresh:
            continue
        if i in usedA or j in usedB:
            continue
        usedA.add(i)
        usedB.add(j)
        matches.append((i, j))

    unmatched_A = [i for i in range(len(setA)) if i not in usedA]
    unmatched_B = [j for j in range(len(setB)) if j not in usedB]
    return matches, unmatched_A, unmatched_B

# ---------- 4) Fusion (average boxes) ----------
# TODO: TO BE REPLACED BY YOUR OWN ALGORITHM

def fuse_average_boxes(o1: Object3d, o2: Object3d) -> Object3d:
    """
    Average (x,y,z), (w,l,h), and yaw (circular mean).
    Copies timestamps/id_frame/classification from o1.
    Uncertainties: simple average of the two vectors if both present; else keep the existing one.
    """
    y1 = quat_to_yaw(o1.orientation)
    y2 = quat_to_yaw(o2.orientation)
    y_mean = circular_mean(np.array([y1, y2], dtype=float))

    fused_unc = None
    if isinstance(o1.uncertainties, list) and isinstance(o2.uncertainties, list) and \
       len(o1.uncertainties) == len(o2.uncertainties):
        fused_unc = list((np.array(o1.uncertainties)+np.array(o2.uncertainties))/2.0)
    else:
        fused_unc = o1.uncertainties or o2.uncertainties

    return Object3d(
        id_frame=o1.id_fame,
        object_id=f"{o1.object_id}|{o2.object_id}",
        timestamp=o1.timestamp,
        xdistance=0.5*(o1.xdistance + o2.xdistance),
        ydistance=0.5*(o1.ydistance + o2.ydistance),
        zdistance=0.5*(o1.zdistance + o2.zdistance),
        width=0.5*(o1.width + o2.width),
        depth=0.5*(o1.depth + o2.depth),
        height=0.5*(o1.height + o2.height),
        orientation=yaw_to_quat(y_mean),
        classification=o1.classification or o2.classification,
        uncertainties=fused_unc,
        dt=o1.dt,
    )

def fuse_matches_avg(setA: list, setB: list, matches: list) -> list:
    """Fuse matched pairs by averaging."""
    fused = []
    for iA, iB in matches:
        fused.append(fuse_average_boxes(setA[iA], setB[iB]))
    return fused


# ---------- 5) Plots ----------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def plot_first_frames_global(
    gt_by_sample,
    noisy_low_by_sample,
    noisy_high_by_sample,
    fused_by_sample=None,
    num_frames=5,
    dist_range=None
):
    """
    Visualize first `num_frames` frames in global coordinates (x,y plane).

    Args:
        gt_by_sample:         dict[sample_token -> list[Object3d]]
        noisy_low_by_sample:  dict[sample_token -> list[Object3d]]
        noisy_high_by_sample: dict[sample_token -> list[Object3d]]
        fused_by_sample:      dict[sample_token -> list[Object3d]] or None
        num_frames:           number of frames to plot
        dist_range:           axis half-range (meters) or None for auto-scaling
    """

    def box_corners_global(obj):
        """Return (4,2) array of box corners in global x-y."""
        cx, cy = obj.xdistance, obj.ydistance
        w, l = obj.width, obj.depth
        yaw = quat_to_yaw(obj.orientation)
        # Local corners
        x_c = np.array([ l/2,  l/2, -l/2, -l/2])
        y_c = np.array([ w/2, -w/2, -w/2,  w/2])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])
        corners = np.stack([x_c, y_c], axis=1) @ R.T + np.array([cx, cy])
        return corners

    sample_tokens = list(gt_by_sample.keys())[:num_frames]
    n = len(sample_tokens)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # --- global bounds ---
    all_xy = []
    for token in sample_tokens:
        for objs in gt_by_sample[token]:
            all_xy.append([objs.xdistance, objs.ydistance])
    all_xy = np.array(all_xy)
    if dist_range is None:
        x_min, y_min = all_xy.min(axis=0)
        x_max, y_max = all_xy.max(axis=0)
        pad = 10.0
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min - pad, y_max + pad)
    else:
        first_objs = gt_by_sample[sample_tokens[0]]
        if len(first_objs):
            cx = np.mean([o.xdistance for o in first_objs])
            cy = np.mean([o.ydistance for o in first_objs])
        else:
            cx, cy = 0, 0
        xlim = (cx - dist_range, cx + dist_range)
        ylim = (cy - dist_range, cy + dist_range)

    for idx, token in enumerate(sample_tokens):
        ax = axes[idx]
        gt_objs = gt_by_sample.get(token, [])
        noisy_low = noisy_low_by_sample.get(token, [])
        noisy_high = noisy_high_by_sample.get(token, [])
        fused = fused_by_sample.get(token, []) if fused_by_sample else []

        # --- Draw ---
        for o in gt_objs:
            ax.add_patch(Polygon(box_corners_global(o), closed=True, fill=False,
                                 edgecolor='blue', linewidth=1.0, label='GT'))
        for o in noisy_low:
            ax.add_patch(Polygon(box_corners_global(o), closed=True, fill=False,
                                 edgecolor='green', linewidth=0.8, label='Low noise'))
        for o in noisy_high:
            ax.add_patch(Polygon(box_corners_global(o), closed=True, fill=False,
                                 edgecolor='red', linewidth=0.8, label='High noise'))
        for o in fused:
            ax.add_patch(Polygon(box_corners_global(o), closed=True, fill=False,
                                 edgecolor='orange', linestyle='--', linewidth=1.2, label='Fused'))

        ax.set_title(f"Frame {idx+1}\n{token[:6]}...", fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel("Global X [m]")
        ax.set_ylabel("Global Y [m]")

    # Legend (only once)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper right', ncol=4, frameon=False)
    fig.suptitle("Top-down global visualization (GT / Noisy / Fused)", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------- Example usage ----------
if __name__ == "__main__":
    # Path to nuScenes json folder (contains sample.json, sample_annotation.json, sample_data.json)
    ann_root = "/dataset/nuscenes/v1.0-trainval" 

    # 1) Load GT
    print("#------------------------------------------")
    print("1 - Loading data from ground truth nuScenes")
    gt_by_sample = load_nuscenes_gt(ann_root, prefer_channel="LIDAR_TOP", fraction=0.2) # loads only 20% of data

    # 2) Create two noisy datasets with different uncertainty levels
    print("#------------------------------------------")
    print("2 - Applying noise to simualte 2 sensors")
    sigmas_low = dict(x=0.20, y=0.20, z=0.20, yaw=np.deg2rad(0.05), w=0.5, l=0.5, h=0.5)
    sigmas_high = dict(x=0.50, y=0.50, z=0.50, yaw=np.deg2rad(0.50), w=1.00, l=1.00, h=1.00)

    noisy_per_sample = make_two_noise_levels(gt_by_sample, sigmas_low, sigmas_high, seed=42)

    # 3) For a given sample, associate low vs high noise and fuse
    print("#------------------------------------------")
    print("3 - Association and Fusion Algorithm")
    fused_results = {}
    noisy_low_by_sample = {}
    noisy_high_by_sample = {}

    for sample_token, (noisy_low, noisy_high) in noisy_per_sample.items():
        # keep the noisy sets per sample (for plotting)
        noisy_low_by_sample[sample_token] = noisy_low
        noisy_high_by_sample[sample_token] = noisy_high

        # associate & fuse
        matches, unA, unB = associate_greedy_2d(noisy_low, noisy_high, dist_thresh=5.0)
        fused = fuse_matches_avg(noisy_low, noisy_high, matches)
        fused_results[sample_token] = fused

        # optional example print
        if fused:
            f0 = fused[0]
            yaw_deg = np.rad2deg(quat_to_yaw(f0.orientation))
            # print(f"[{sample_token[:6]}] matches={len(matches)} unA={len(unA)} unB={len(unB)} "
            #       f"fused0: x={f0.xdistance:.2f}, y={f0.ydistance:.2f}, z={f0.zdistance:.2f}, "
            #       f"w={f0.width:.2f}, l={f0.depth:.2f}, h={f0.height:.2f}, yaw={yaw_deg:.1f}°)")

    print("#------------------------------------------")
    print("4 - Plot results")

    # If your GT dict is named objs_by_sample, alias it:
    # gt_by_sample = objs_by_sample

    plot_first_frames_global(
        gt_by_sample=gt_by_sample,
        noisy_low_by_sample=noisy_low_by_sample,
        noisy_high_by_sample=noisy_high_by_sample,
        fused_by_sample=fused_results,   # pass precomputed fused boxes
        num_frames=2,
        dist_range=100
    )

