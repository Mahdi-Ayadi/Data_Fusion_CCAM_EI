# Data Fusion Project

This repository contains a compact research-style pipeline for experimenting with multi-sensor 3D detection fusion on nuScenes. The work follows the brief provided in *TP_centrale.pdf*, implementing and comparing three late-fusion strategies (plus a baseline) using synthetic sensor noise generated from ground truth boxes.

---

## Contents

```
├── data_fusion/           # Python package with reusable modules
│   ├── models.py          # Quaternion fallback + Object3d container
│   ├── data_loading.py    # nuScenes JSON loader + synthetic noise generators
│   ├── association.py     # Euclidean and Mahalanobis matching
│   ├── fusion.py          # Baseline + Weighted LS fusion routines
│   ├── late_fusion.py     # Learnable attention gate (heuristic fallback)
│   ├── tracking.py        # Kalman filter tracker for sensor fusion
│   ├── evaluation.py      # Error metrics and nuScenes-style scoring
│   └── plotting.py        # Matplotlib helpers producing top-down figures
├── dataset/               # Expected nuScenes annotations (sample/sample_data/sample_annotation)
├── figures/               # Auto-generated qualitative plots per fusion method
├── models/                # Saved late-fusion checkpoints (ignored by git)
├── fusion_centrale.py     # Main script running all fusion pipelines
├── train_late_fusion.py   # Training entry point for the IA-inspired late fusion network
├── requirements.txt       # Python dependencies
└── TP_centrale.pdf        # Project brief / instructions
```

---

## Data Setup

The project assumes access to the nuScenes annotation JSONs (train/val). Copy the files into:

```
Data_Fusion_CCAM_EI/dataset/nuscenes/v1.0-trainval/
 ├─ sample.json
 ├─ sample_data.json
 └─ sample_annotation.json
```

Alternatively, set the `NUSCENES_ANN_ROOT` environment variable to point at an existing nuScenes directory.

---

## Installation

1. Create and activate a Python environment (conda or venv).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For training the late-fusion network, PyTorch (CPU or GPU build) is required; it is already listed in `requirements.txt`.

---

## Running the Fusion Demo

Execute the orchestrator to simulate sensors, fuse detections, and evaluate each method:

```bash
python3 fusion_centrale.py
```

The script will:

1. Load ground-truth boxes (subsampled via `fraction=0.05` for speed).
2. Generate two noisy detection sets mimicking LiDAR and camera outputs (with uncertainty vectors retained).
3. Run four fusion pipelines:
   - **Baseline**: Euclidean matching + simple averaging.
   - **Weighted Least Squares**: Uncertainty-aware matching and inverse-variance fusion (project option 1).
   - **Kalman Fusion**: Constant-velocity tracker combining both sensors sequentially (option 2).
   - **Late Fusion (IA-inspired)**: Attention gate weighting sensors per match; uses a learned model if available (option 3).
4. Print per-method error statistics and nuScenes-style metrics.
5. Save qualitative plots in `figures/` showing GT, noisy detections, and fused boxes.

If a trained late-fusion checkpoint is found at `models/late_fusion.pth`, it will be loaded automatically. Otherwise, the IA fusion falls back to a heuristic gate.

---

## Training the IA Late-Fusion Gate

The repository includes `train_late_fusion.py`, which learns a lightweight MLP that predicts sensor weights from reliability cues and disagreement features.

Run it with:

```bash
python3 train_late_fusion.py \
    --ann-root /path/to/nuscenes/v1.0-trainval \
    --fraction 0.2 \
    --epochs 30 \
    --batch-size 256 \
    --lr 1e-3 \
    --output models/late_fusion.pth
```

Key notes:

- The script creates synthetic sensor pairs the same way as the demo, then builds training triples `(features, sensor1, sensor2, ground_truth)` from matched detections.
- The network is tiny (1.3 k parameters: 6→32→32→1). It trains quickly on CPU but can leverage GPU if available.
- A tqdm progress bar displays batch progress for both train and validation loops.
- The best-performing checkpoint is written to `models/late_fusion.pth`, ready for `fusion_centrale.py` to consume.

---

## Metrics & Outputs

- **Console metrics**: For each method, the script prints simple aggregate errors (position, yaw, size) and nuScenes-style mAP, mATE, mASE, mAOE (with synthetic detections all treated as score=1). These numbers let you compare recall/precision-style performance across fusion techniques.
- **Figures**: Saved under `figures/` for quick qualitative inspection.
- **Checkpoints**: Trained late-fusion models reside in `models/` (git-ignored by default).

---

## Extending the Project

The current pipeline uses synthetic noise around ground truth, which makes it ideal for rapid experimentation. To adapt it to real detector outputs:

1. Replace `make_two_noise_levels` with loaders that ingest actual LiDAR and camera prediction PKL files.
2. Feed detector-specific uncertainties (if available) into each `Object3d` before fusion.
3. Adjust association thresholds and Kalman noise parameters to reflect the real sensor characteristics.
4. Expand the evaluation module to integrate “true” nuScenes metrics (NDS) if you integrate the official nuScenes evaluation toolkit.

The modular design – especially the `data_fusion` package – supports these changes with minimal disruption.

---

## References

- Maryem Fadili et al., “Weighted least-squares multi-detection fusion and Kalman filter-based tracking for collaborative perception systems.” (2025)
- Maryem Fadili et al., “Evaluation of an Uncertainty-Aware Late Fusion Algorithm…” (2025)
- Lennart L. F. Jahn et al., “Enhancing lane detection with a lightweight collaborative late fusion model.” Robotics and Autonomous Systems, 2024.
- Holger Caesar et al., “nuScenes: A multimodal dataset for autonomous driving.” CVPR 2020.

---

## Quick Commands

```bash
# install deps
pip install -r requirements.txt

# run fusion demo
python3 fusion_centrale.py

# train late-fusion gate (optional)
python3 train_late_fusion.py --fraction 0.2
```

Happy experimenting!
