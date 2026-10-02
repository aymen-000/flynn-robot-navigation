from __future__ import annotations


from src.flynn.core.utils import wrap_pi

import numpy as np
import math
from typing import List, Optional
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class SearchNode:
    f_score: float
    g_score: float = field(compare=False)
    x: float = field(compare=False)
    y: float = field(compare=False)
    yaw: float = field(compare=False)
    depth: int = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False, default=None)

class VFHPlusPlanner:
    def __init__(self, 
                 arena_half_extent: float,
                 robot_radius: float = 0.2,
                 safety_dist: float = 0.1,
                 active_window: float = 3, # meters, lookahead window
                 sector_angle: float = 2.0, # degrees
                 u_max: float = 10.0, # max certainty for histogram
                 wide_opening_width: int = 5, # number of sectors for wide opening
                 search_depth: int = 5, # A* search depth
                 step_size: float = 0.6, # A* step size
                 goal_threshold: float = 0.3): # distance threshold for goal reached
        
        self.arena_half_extent = float(arena_half_extent)
        self.goal_threshold = float(goal_threshold)
        self.robot_radius = robot_radius
        self.safety_dist = safety_dist
        self.active_window = active_window
        self.alpha = float(np.deg2rad(sector_angle))
        self.num_sectors = int(np.ceil(2 * np.pi / self.alpha))
        self.u_max = u_max
        self.wide_opening_width = wide_opening_width
        
        self.primary_hist = np.zeros(self.num_sectors, dtype=np.float32)
        self.binary_hist = np.zeros(self.num_sectors, dtype=bool)
        
        self.mu1 = 10.0 # Target direction weight
        self.mu2 = 4.0 # Current direction weight (alignment with robot heading)
        self.mu3 = 10.0 # Previous direction weight (hysteresis) - increased to reduce jumping
        self.mu4 = 15.0 # Path consistency weight (deviation from previous path direction)
        
        # Direction stabilization parameters
        self.smoothing_factor = 0.3  # Exponential smoothing (lower = smoother, 0.1-0.5 range)
        self.max_angular_rate = np.deg2rad(5.0)  # Relaxed for search, we penalize in cost
        
        self.prev_steering_dir = 0.0
        self.prev_path_dir = None  # Direction to first waypoint of previous path
        
        # A* params
        self.search_depth = search_depth
        self.step_size = step_size

    def _update_histograms(self, start_xy: np.ndarray, obstacles_xy: np.ndarray, obstacle_radius: float,
                           current_yaw: float = None, fov: float = None):

        self.primary_hist.fill(0.0)
        
        if len(obstacles_xy) == 0:
            return

        # Vectorized distance calculation
        dx = obstacles_xy[:, 0] - start_xy[0]
        dy = obstacles_xy[:, 1] - start_xy[1]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq)

        # Filter by active window (circular)
        mask = dist <= self.active_window
        
        # Filter by FOV if provided
        if current_yaw is not None and fov is not None:
            angles = np.arctan2(dy, dx)
            angle_diff = np.abs(wrap_pi(angles - current_yaw))
            fov_rad = np.deg2rad(fov) / 2.0
            mask = mask & (angle_diff <= fov_rad)
        
        valid_dx = dx[mask]
        valid_dy = dy[mask]
        valid_dist = dist[mask]
        
        safe_r = self.robot_radius + self.safety_dist + obstacle_radius

        sec_angles = np.arange(self.num_sectors) * self.alpha
        
        for i in range(len(valid_dist)):
            d = valid_dist[i]
            beta = math.atan2(valid_dy[i], valid_dx[i])
            
            if d < safe_r:
                enlargement_angle = np.pi/2 # Block full circle
            else:
                enlargement_angle = math.asin(safe_r / d)
            
            m_val = (1.0 - (d / self.active_window)) * self.u_max
            if m_val < 0: m_val = 0.0

            diffs = np.abs(np.arctan2(np.sin(sec_angles - beta), np.cos(sec_angles - beta)))
            sector_mask = diffs <= enlargement_angle
            self.primary_hist[sector_mask] += m_val

        A = self.arena_half_extent - 0.1
        
        for k in range(self.num_sectors):
            angle = k * self.alpha
            # Ray direction
            rx = math.cos(angle)
            ry = math.sin(angle)
            
            dist_to_wall = float('inf')
            
            if rx > 1e-3:
                d = (A - start_xy[0]) / rx
                if d >= 0: dist_to_wall = min(dist_to_wall, d)
            # 2. x = -A (Left wall)
            if rx < -1e-3:
                 d = (-A - start_xy[0]) / rx
                 if d >= 0: dist_to_wall = min(dist_to_wall, d)
            # 3. y = A (Top wall)
            if ry > 1e-3:
                d = (A - start_xy[1]) / ry
                if d >= 0: dist_to_wall = min(dist_to_wall, d)
            # 4. y = -A (Bottom wall)
            if ry < -1e-3:
                d = (-A - start_xy[1]) / ry
                if d >= 0: dist_to_wall = min(dist_to_wall, d)
                
            if dist_to_wall <= self.active_window:
                 m_val = (1.0 - (dist_to_wall / self.active_window)) * self.u_max
                 if m_val < 0: m_val = 0.0
                 self.primary_hist[k] += m_val

    def _binary_polar_histogram(self, threshold: float):

        limit = threshold
        self.binary_hist = self.primary_hist > limit
        
    def _find_candidate_directions(self, target_dir: float) -> List[float]:
        # Identify valleys (sequences of free sectors)
        candidates = []
  
        is_free = ~self.binary_hist
        
        valleys = []
        if np.all(is_free):
            valleys.append((0, self.num_sectors + 0)) # Full circle
        elif np.any(is_free):
             extended = np.concatenate([is_free, is_free])
             
             for i in range(self.num_sectors):
                 prev_idx = (i - 1) % self.num_sectors  # Proper wraparound
                 if not is_free[prev_idx] and is_free[i]:
                     # Found start of a valley at i
                     # Trace forward with safety limit
                     j = i
                     max_trace = 2 * self.num_sectors  # Safety limit
                     while j < max_trace and extended[j]:
                         j += 1
                     # Valley is [i, j-1] (modulo num_sectors effectively)
                     valleys.append((i, j-1))

        for start, end in valleys:
            width = end - start + 1

            if width > self.wide_opening_width:

                target_idx = target_dir / self.alpha

                
                s_angle = wrap_pi(start * self.alpha)
                e_angle = wrap_pi(end * self.alpha) # careful, end might have wrapped
                
    
                mid_angle = wrap_pi((s_angle + e_angle) / 2.0) # Careful with wrapping averagin
                t_idx = (wrap_pi(target_dir) if wrap_pi(target_dir) >= 0 else wrap_pi(target_dir) + 2*np.pi) / self.alpha
                

                cand_angles = []
                
                k_l = start + min(width//2, self.wide_opening_width//2) # somewhat inside
                ang_l = k_l * self.alpha
                cand_angles.append(ang_l)
                
                k_r = end - min(width//2, self.wide_opening_width//2)
                ang_r = k_r * self.alpha
                cand_angles.append(ang_r)
                

                for c in cand_angles:
                    candidates.append(wrap_pi(c))
                t_k = int((wrap_pi(target_dir) % (2*np.pi)) / self.alpha)
                if is_free[t_k]:
                    candidates.append(target_dir)
                    
            else:
                center_idx = (start + end) / 2.0
                candidates.append(wrap_pi(center_idx * self.alpha))
                
        
        return candidates

    def _astar_search(self, start_xy: np.ndarray, goal_xy: np.ndarray, 
                      obstacles_xy: np.ndarray, obstacle_radius: float, 
                      start_yaw: float) -> Optional[List[np.ndarray]]:
        
        start_node = SearchNode(
            f_score=0.0, g_score=0.0,
            x=start_xy[0], y=start_xy[1], yaw=start_yaw,
            depth=0, parent=None
        )
        
        open_set = [start_node]
        heapq.heapify(open_set)
        
        closest_node = start_node
        min_dist = float('inf')
        
        max_nodes = 300 # Safety limit
        count = 0
        
        closed_set = set()

        dist_to_goal = math.hypot(goal_xy[0] - start_node.x, goal_xy[1] - start_node.y)
        search_dist = min(self.search_depth * self.step_size, dist_to_goal)
        search_depth = max(1, int(search_dist / self.step_size))
        
        max_candidates = 5
        
        while open_set and count < max_nodes:
            count += 1
            current = heapq.heappop(open_set)
            
            state_key = (round(current.x, 2), round(current.y, 2), round(current.yaw, 2))
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            dist_to_goal = math.hypot(goal_xy[0] - current.x, goal_xy[1] - current.y)
            if dist_to_goal < min_dist:
                min_dist = dist_to_goal
                closest_node = current
            

            if dist_to_goal < self.goal_threshold or current.depth >= search_depth:
                return self._reconstruct_path(current)

            pos = np.array([current.x, current.y])
            self._update_histograms(pos, obstacles_xy, obstacle_radius)
            self._binary_polar_histogram(threshold=0)
            
  
            dx = goal_xy[0] - current.x
            dy = goal_xy[1] - current.y
            target_dir = math.atan2(dy, dx)
            
            candidates = self._find_candidate_directions(target_dir)
            
            if not candidates:
                continue
            
            scored_candidates = []
            for cand_dir in candidates:
                d_target = abs(wrap_pi(cand_dir - target_dir))
                d_yaw = abs(wrap_pi(cand_dir - current.yaw))
                score = self.mu1 * d_target + self.mu2 * d_yaw
                scored_candidates.append((score, cand_dir))
            scored_candidates.sort(key=lambda x: x[0])
            candidates = [c[1] for c in scored_candidates[:max_candidates]]
                
            for cand_dir in candidates:
                next_x = current.x + self.step_size * math.cos(cand_dir)
                next_y = current.y + self.step_size * math.sin(cand_dir)
                
                child_key = (round(next_x, 2), round(next_y, 2), round(cand_dir, 2))
                if child_key in closed_set:
                    continue
                
                d_yaw = abs(wrap_pi(cand_dir - current.yaw))
                
                d_target = abs(wrap_pi(cand_dir - target_dir))
                
                prev_yaw = current.parent.yaw if current.parent else current.yaw
                d_prev = abs(wrap_pi(cand_dir - prev_yaw))
                
                if current.depth == 0 and self.prev_path_dir is not None:
                    d_path = abs(wrap_pi(cand_dir - self.prev_path_dir))
                else:
                    d_path = 0.0

                edge_cost = (
                    self.step_size + 
                    self.mu1 * d_target * 0.1 + 
                    self.mu2 * d_yaw * 0.1 +
                    self.mu3 * d_prev * 0.1 +  # Hysteresis cost
                    self.mu4 * d_path * 0.1    # Path consistency cost
                )
                
                new_g = current.g_score + edge_cost
                
                h = math.hypot(goal_xy[0] - next_x, goal_xy[1] - next_y)
                
                new_f = new_g + h
                
                child = SearchNode(
                    f_score=new_f, g_score=new_g,
                    x=next_x, y=next_y, yaw=cand_dir,
                    depth=current.depth + 1,
                    parent=current
                )
                heapq.heappush(open_set, child)
                
        # Return best path found if loop finishes
        return self._reconstruct_path(closest_node)

    def _reconstruct_path(self, node: SearchNode) -> List[np.ndarray]:
        path = []
        curr = node
        while curr is not None:
            path.append(np.array([curr.x, curr.y], dtype=np.float32))
            curr = curr.parent
        return path[::-1] # Reverse

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray, obstacles_xy: np.ndarray, 
             obstacle_radius: float, current_yaw: float, fov: float = 160.0) -> Optional[List[np.ndarray]]:
        """
        VFH* local planning step (A* Lookahead).
        Returns a path: [start_xy, wp1, wp2, ...].
        """
        path = self._astar_search(start_xy, goal_xy, obstacles_xy, obstacle_radius, self.prev_steering_dir)
        
        if path is None or len(path) < 2:
            self._update_histograms(start_xy, obstacles_xy, obstacle_radius, current_yaw, fov)
            self._binary_polar_histogram(threshold=0)
            candidates = self._find_candidate_directions(current_yaw)
            if candidates:
                best_dir = candidates[0] # Pick first
                wx = start_xy[0] + self.step_size * math.cos(best_dir)
                wy = start_xy[1] + self.step_size * math.sin(best_dir)
                path = [start_xy, np.array([wx, wy], dtype=np.float32)]
            else:
                return [start_xy, start_xy] # Stop
        
        if len(path) > 1:
            p0 = path[0]
            p1 = path[1]
            raw_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            
            # Store raw path direction for consistency cost in next call
            self.prev_path_dir = raw_yaw
            
            # Apply exponential smoothing
            smoothed_yaw = wrap_pi(
                self.prev_steering_dir + 
                self.smoothing_factor * wrap_pi(raw_yaw - self.prev_steering_dir)
            )
            
            # Apply rate limiting
            delta = wrap_pi(smoothed_yaw - self.prev_steering_dir)
            if abs(delta) > self.max_angular_rate:
                delta = np.sign(delta) * self.max_angular_rate
            
            self.prev_steering_dir = wrap_pi(self.prev_steering_dir + delta)

        return path
