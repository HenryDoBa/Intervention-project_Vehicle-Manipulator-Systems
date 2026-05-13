"""
bezier.py
Pure-function helpers for quadratic and cubic Bézier curves.
No ROS / OpenCV dependencies — just NumPy.
"""

import numpy as np


def bezier(t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Position on quadratic Bézier at parameter t ∈ [0, 1]."""
    return (1.0 - t)**2 * P0 + 2.0 * (1.0 - t) * t * P1 + t**2 * P2


def bezier_vel(t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Tangent (derivative w.r.t. t) of quadratic Bézier."""
    return 2.0 * (1.0 - t) * (P1 - P0) + 2.0 * t * (P2 - P1)


def bezier_cubic(t: float,
                 P0: np.ndarray, P1: np.ndarray,
                 P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
    """Position on cubic Bézier at parameter t ∈ [0, 1]."""
    u = 1.0 - t
    return u**3 * P0 + 3.0 * u**2 * t * P1 + 3.0 * u * t**2 * P2 + t**3 * P3


def bezier_cubic_vel(t: float,
                     P0: np.ndarray, P1: np.ndarray,
                     P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
    """Tangent (derivative w.r.t. t) of cubic Bézier."""
    u = 1.0 - t
    return 3.0 * (u**2 * (P1 - P0) + 2.0 * u * t * (P2 - P1) + t**2 * (P3 - P2))