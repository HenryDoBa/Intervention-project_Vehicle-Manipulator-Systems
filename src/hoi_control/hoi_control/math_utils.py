"""
math_utils.py
Pure-math helper functions with no ROS / OpenCV dependencies.
"""

import math


def angle_wrap(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi