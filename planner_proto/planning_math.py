# planner_proto/planning_math.py - сюди переноситься nn_route, one_insertion_improve, dist, centroid,...

from __future__ import annotations
from typing import List, Tuple
import math


Point = Tuple[float, float]


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Звичайна евклідова відстань у пікселях."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def nearest_neighbor_order(points: List[Point], start_index: int = 0) -> List[int]:
    """
    Примітивний алгоритм "найближчого сусіда":
    повертає порядок обходу індексів точок.
    """
    n = len(points)
    if n == 0:
        return []

    unvisited = set(range(n))
    order: List[int] = []

    current = start_index if 0 <= start_index < n else 0

    while unvisited:
        if current not in unvisited:
            # якщо стартовий вже вийшов — беремо довільний з тих, що лишились
            current = next(iter(unvisited))

        unvisited.remove(current)
        order.append(current)

        if not unvisited:
            break

        cx, cy = points[current]
        # знаходимо найближчу ще не відвідану точку
        current = min(
            unvisited,
            key=lambda i: euclidean_distance((cx, cy), points[i])
        )

    return order
