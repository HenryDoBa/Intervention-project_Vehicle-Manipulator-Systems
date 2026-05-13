"""
camera_intrinsics.py
Camera intrinsics derived from image size and horizontal FOV
(from turtlebot_featherstone.scn).

Kept separate from config.py because these are computed values, not tunables.
"""

import math
import numpy as np


_IMG_W, _IMG_H = 1920, 1080
_HFOV_DEG      = 69.0
_FX = _FY      = (_IMG_W / 2.0) / math.tan(math.radians(_HFOV_DEG / 2.0))
_CX, _CY       = _IMG_W / 2.0, _IMG_H / 2.0

CAMERA_MATRIX  = np.array([[_FX, 0.0, _CX],
                           [0.0, _FY, _CY],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_COEFFS    = np.zeros(5, dtype=np.float64)  # simulation: no distortion