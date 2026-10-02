from __future__ import annotations


from src.flynn.core.utils import wrap_pi

import numpy as np
import math


class PIDState:
    def __init__(self):
        self.integral = 0.0
        self.last_error = 0.0

def pid_line_follower(
    xy: np.ndarray,          # Current [x, y]
    yaw: float,             # Current heading
    start_xy: np.ndarray,    # Point A (where the segment started)
    target_wp: np.ndarray,   # Point B (the waypoint)
    state: PIDState,         # Persistent PID memory
    dt: float = 0.02          # Time step
) -> np.ndarray:
    if start_xy is None:
        start_xy = xy
    if state is None:
        state = PIDState()

    # 1. Calculate Vectors
    path_vec = target_wp - start_xy
    robot_vec = xy - start_xy
    
    path_len = np.linalg.norm(path_vec) + 1e-6
    path_unit_vec = path_vec / path_len


    cte = (path_unit_vec[0] * robot_vec[1]) - (path_unit_vec[1] * robot_vec[0])

    target_yaw = math.atan2(path_vec[1], path_vec[0])
    heading_err = wrap_pi(target_yaw - yaw)

    v_fwd = 0.7

    heading_err = wrap_pi(target_yaw - yaw)

    k_cte = 1.0
    cte_correction = math.atan2(k_cte * -cte, max(v_fwd, 0.1))

    steering_angle = wrap_pi(heading_err + cte_correction)

    return np.array([v_fwd, steering_angle], dtype=np.float32)
