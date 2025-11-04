from __future__ import annotations

from math import hypot, sqrt
from typing import List, Tuple

import numpy as np

from .math_utils import MIN_VARIANCE, sigma_from_uncertainties
from .models import Object3d


def associate_greedy_euclidean(
    setA: List[Object3d],
    setB: List[Object3d],
    dist_thresh: float = 5.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Baseline greedy nearest-neighbour association in Euclidean (x, y) space."""
    if not setA or not setB:
        return [], list(range(len(setA))), list(range(len(setB)))

    A_xy = np.array([[o.xdistance, o.ydistance] for o in setA], dtype=float)
    B_xy = np.array([[o.xdistance, o.ydistance] for o in setB], dtype=float)

    usedA = set()
    usedB = set()
    matches: List[Tuple[int, int]] = []
    pairs = []

    for i in range(len(setA)):
        for j in range(len(setB)):
            d = hypot(A_xy[i, 0] - B_xy[j, 0], A_xy[i, 1] - B_xy[j, 1])
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


def associate_weighted_mahalanobis(
    setA: List[Object3d],
    setB: List[Object3d],
    dist_thresh: float = 3.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Weighted association using a Mahalanobis-like distance in (x, y)."""
    if not setA or not setB:
        return [], list(range(len(setA))), list(range(len(setB)))

    pairs = []
    for i, obj_a in enumerate(setA):
        for j, obj_b in enumerate(setB):
            dx = obj_a.xdistance - obj_b.xdistance
            dy = obj_a.ydistance - obj_b.ydistance

            sx = sigma_from_uncertainties(obj_a, 0) ** 2 + sigma_from_uncertainties(obj_b, 0) ** 2
            sy = sigma_from_uncertainties(obj_a, 1) ** 2 + sigma_from_uncertainties(obj_b, 1) ** 2

            sx = max(sx, MIN_VARIANCE)
            sy = max(sy, MIN_VARIANCE)

            maha_dist = sqrt((dx * dx) / sx + (dy * dy) / sy)
            pairs.append((maha_dist, i, j))

    pairs.sort(key=lambda x: x[0])

    usedA = set()
    usedB = set()
    matches: List[Tuple[int, int]] = []

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
