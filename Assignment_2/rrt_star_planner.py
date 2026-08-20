"""
Assignment #2 Template file
"""

import random
import math
import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees* (RRT*)
for the problem setup provided by the RRT_dubins_problem class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file: rrt_star_planner.py. Your
   implementation can be tested by running dubins_path_problem.py (check the
   main function).
2. Read all class and function documentation in dubins_path_problem.py carefully.
   There are plenty of helper functions in the class to ease implementation.
3. Your solution must meet all the conditions specified below.
4. Below are some DOs and DONTs for this problem.

Conditions
-------------------
There are several conditions that must be satisfied for an acceptable solution.
These may or may not be verified by the auto-grading script.

1. The solution loop must not run for more than a certain number of random iterations
   (specified by the class member max_iter). This is mainly a safety
   measure to avoid time-out-related issues and will be set generously.
2. The planning function must return a list of nodes that represent a collision-free path
   from the start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must define a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation for the Node class to understand the terminology).
3. The returned path should have the start node at index 0 and the goal node at index -1.
   The parent node for node i from the list should be node i-1 from the list (i.e.,
   the path should be a valid, continuous list of connected nodes).
4. The node locations must not lie outside the map boundaries specified by
   RRT_dubins_problem.map_area.

DOs and DONTs
-------------------
1. DO NOT rename the file rrt_star_planner.py for submission.
2. DO NOT change the rrt_star_planner function signature.
3. DO NOT import anything other than what is already imported in this file.
4. YOU MAY write additional helper functions in this file to reduce code repetition,
   but these functions can only be used inside the rrt_star_planner function
   (since only the rrt_star_planner function will be imported for grading).
"""


def rrt_star_planner(rrt_dubins, display_map=False):
   """
   Execute RRT* planning using Dubins-style paths. Make sure to populate the node_list.

   Inputs
   -------------
   rrt_dubins  - (RRT_dubins_problem) Class containing the planning
               problem specification.
   display_map - (boolean) Flag for animation on or off (OPTIONAL).

   Outputs
   --------------
   (list of Node) This must be a valid list of connected nodes that form
                  a continuous path from the start node to the goal node.

   NOTE: In order for the rrt_dubins.draw_graph function to work properly, it is
   important to populate rrt_dubins.node_list with all valid RRT nodes.
   """

   # Helper: fast Dubins path length (cost only, no path generation)
   # Inlines the six Dubins word formulas (LSL, RSR, LSR, RSL, RLR, LRL)
   # and returns only the shortest path length, skipping the expensive
   # discretised path that dubins_path_planning.dubins_path_planning builds.
   # This replaces calc_new_cost for the best-parent and rewire loops.
   _floor = math.floor
   _sin   = math.sin
   _cos   = math.cos
   _atan2 = math.atan2
   _sqrt  = math.sqrt
   _acos  = math.acos
   _hypot = math.hypot
   _pi    = math.pi
   _2pi   = 2.0 * math.pi
   _inf   = float("inf")

   def _mod2pi(theta):
      return theta - _2pi * _floor(theta / _2pi)

   def dubins_cost_only(sx, sy, syaw, ex, ey, eyaw, curvature):
      dx = ex - sx
      dy = ey - sy
      D = _hypot(dx, dy)
      d = D * curvature

      theta = _mod2pi(_atan2(dy, dx))
      alpha = _mod2pi(syaw - theta)
      beta  = _mod2pi(eyaw - theta)

      sa = _sin(alpha)
      sb = _sin(beta)
      ca = _cos(alpha)
      cb = _cos(beta)
      c_ab = _cos(alpha - beta)

      best = _inf

      # LSL
      p_sq = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sa - sb)
      if p_sq >= 0.0:
         tmp1 = _atan2(cb - ca, d + sa - sb)
         t = _mod2pi(-alpha + tmp1)
         p = _sqrt(p_sq)
         q = _mod2pi(beta - tmp1)
         c = t + p + q
         if c < best:
            best = c

      # RSR
      p_sq = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sb - sa)
      if p_sq >= 0.0:
         tmp1 = _atan2(ca - cb, d - sa + sb)
         t = _mod2pi(alpha - tmp1)
         p = _sqrt(p_sq)
         q = _mod2pi(-beta + tmp1)
         c = t + p + q
         if c < best:
            best = c

      # LSR
      p_sq = -2.0 + d * d + 2.0 * c_ab + 2.0 * d * (sa + sb)
      if p_sq >= 0.0:
         p = _sqrt(p_sq)
         tmp2 = _atan2(-ca - cb, d + sa + sb) - _atan2(-2.0, p)
         t = _mod2pi(-alpha + tmp2)
         q = _mod2pi(-_mod2pi(beta) + tmp2)
         c = t + p + q
         if c < best:
            best = c

      # RSL
      p_sq = d * d - 2.0 + 2.0 * c_ab - 2.0 * d * (sa + sb)
      if p_sq >= 0.0:
         p = _sqrt(p_sq)
         tmp2 = _atan2(ca + cb, d - sa - sb) - _atan2(2.0, p)
         t = _mod2pi(alpha - tmp2)
         q = _mod2pi(beta - tmp2)
         c = t + p + q
         if c < best:
            best = c

      # RLR
      tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
      if abs(tmp_rlr) <= 1.0:
         p = _mod2pi(_2pi - _acos(tmp_rlr))
         t = _mod2pi(alpha - _atan2(ca - cb, d - sa + sb) + _mod2pi(p / 2.0))
         q = _mod2pi(alpha - beta - t + _mod2pi(p))
         c = t + p + q
         if c < best:
            best = c

      # LRL
      tmp_lrl = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (-sa + sb)) / 8.0
      if abs(tmp_lrl) <= 1.0:
         p = _mod2pi(_2pi - _acos(tmp_lrl))
         t = _mod2pi(-alpha - _atan2(ca - cb, d + sa - sb) + p / 2.0)
         q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
         c = t + p + q
         if c < best:
            best = c

      return best / curvature

   # Helper: after rewiring a node, propagate the cost delta to all descendants
   def propagate_cost_to_descendants(updated_node, node_list):
      for node in node_list:
         if node.parent is not updated_node:
            continue
         edge_cost = node.cost - node.parent.cost
         node.cost = updated_node.cost + edge_cost
         propagate_cost_to_descendants(node, node_list)

   # Helper: check that the entire Dubins path stays within map boundaries
   def is_path_within_bounds(node, x_min, x_max, y_min, y_max):
      if node is None:
         return False
      for path_x_val in node.path_x:
         if path_x_val < x_min or path_x_val > x_max:
            return False
      for path_y_val in node.path_y:
         if path_y_val < y_min or path_y_val > y_max:
            return False
      return True

   GOAL_BIAS = 0.1
   curvature = rrt_dubins.curvature
   x_min = rrt_dubins.x_lim[0]
   x_max = rrt_dubins.x_lim[1]
   y_min = rrt_dubins.y_lim[0]
   y_max = rrt_dubins.y_lim[1]
   best_goal_node = None

   # Parallel coordinate lists for vectorised nearest-neighbor search
   coords_x = [rrt_dubins.start.x]
   coords_y = [rrt_dubins.start.y]

   # Loop for max iterations
   for i in range(rrt_dubins.max_iter):

      # Adaptive radius: shrinks as tree grows via O(log n / n)^0.5 scaling
      num_nodes = len(rrt_dubins.node_list)
      search_radius = min(
         10.0,
         15.0 * (math.log(num_nodes + 1) / (num_nodes + 1)) ** 0.5
      )
      search_radius_sq = search_radius * search_radius

      # 1. Generate a random vehicle state (x, y, yaw)
      # 1.1 With probability GOAL_BIAS, sample the goal state directly
      if random.random() < GOAL_BIAS:
         random_state = rrt_dubins.Node(
            rrt_dubins.goal.x,
            rrt_dubins.goal.y,
            rrt_dubins.goal.yaw
         )
      # 1.2 Otherwise, sample uniformly within map bounds
      else:
         random_state = rrt_dubins.Node(
            random.uniform(rrt_dubins.x_lim[0], rrt_dubins.x_lim[1]),
            random.uniform(rrt_dubins.y_lim[0], rrt_dubins.y_lim[1]),
            random.uniform(-math.pi, math.pi)
         )

      rx = random_state.x
      ry = random_state.y

      # 2. Find the nearest existing node (vectorised Euclidean distance)
      arr_x = np.array(coords_x)
      arr_y = np.array(coords_y)
      diffs_sq = (arr_x - rx) ** 2 + (arr_y - ry) ** 2
      nearest_idx = int(np.argmin(diffs_sq))
      nearest_node = rrt_dubins.node_list[nearest_idx]

      # 3. Choose best parent from near nodes (RRT* best-parent selection)
      # 3.1 Collect near node indices within adaptive search radius
      near_indices = np.flatnonzero(diffs_sq <= search_radius_sq)

      # 3.2 Evaluate cost through each candidate parent
      #     Uses the fast dubins_cost_only with Euclidean lower-bound pruning
      best_parent = nearest_node
      best_cost = nearest_node.cost + dubins_cost_only(
         nearest_node.x, nearest_node.y, nearest_node.yaw,
         rx, ry, random_state.yaw, curvature)

      for idx in near_indices:
         candidate = rrt_dubins.node_list[idx]
         # 3.2.1 Euclidean lower-bound prune: dubins length >= Euclidean distance
         euclidean_lb = candidate.cost + math.sqrt(diffs_sq[idx])
         if euclidean_lb >= best_cost:
            continue
         # 3.2.2 Compute exact Dubins cost
         candidate_cost = candidate.cost + dubins_cost_only(
            candidate.x, candidate.y, candidate.yaw,
            rx, ry, random_state.yaw, curvature)
         if candidate_cost < best_cost:
            temp_node = rrt_dubins.propagate(candidate, random_state)
            if rrt_dubins.check_collision(temp_node) and is_path_within_bounds(temp_node, x_min, x_max, y_min, y_max):
               best_cost = candidate_cost
               best_parent = candidate

      # 4. Propagate Dubins path from best parent to random state
      new_node = rrt_dubins.propagate(best_parent, random_state)

      # 5. Collision check, boundary check, and node insertion
      if rrt_dubins.check_collision(new_node) and is_path_within_bounds(new_node, x_min, x_max, y_min, y_max):
         rrt_dubins.node_list.append(new_node)
         coords_x.append(new_node.x)
         coords_y.append(new_node.y)

         # 5.1 Rewire near nodes through new_node if cheaper
         for idx in near_indices:
            near_node = rrt_dubins.node_list[idx]
            # 5.1.1 Euclidean lower-bound prune
            euclidean_lb = new_node.cost + math.sqrt(
               (new_node.x - near_node.x) ** 2 + (new_node.y - near_node.y) ** 2)
            if euclidean_lb >= near_node.cost:
               continue
            # 5.1.2 Compute exact Dubins cost
            rewire_cost = new_node.cost + dubins_cost_only(
               new_node.x, new_node.y, new_node.yaw,
               near_node.x, near_node.y, near_node.yaw, curvature)
            if rewire_cost < near_node.cost:
               rewired = rrt_dubins.propagate(new_node, near_node)
               if rrt_dubins.check_collision(rewired) and is_path_within_bounds(rewired, x_min, x_max, y_min, y_max):
                  near_node.parent = rewired.parent
                  near_node.cost = rewired.cost
                  near_node.path_x = rewired.path_x
                  near_node.path_y = rewired.path_y
                  near_node.path_yaw = rewired.path_yaw
                  # 5.2 Propagate cost updates to descendant nodes
                  propagate_cost_to_descendants(near_node, rrt_dubins.node_list)

         # 5.3 Attempt to connect new_node to the goal; track best solution
         goal_node = rrt_dubins.propagate(new_node, rrt_dubins.goal)
         if rrt_dubins.check_collision(goal_node) and is_path_within_bounds(goal_node, x_min, x_max, y_min, y_max):
            if best_goal_node is None or goal_node.cost < best_goal_node.cost:
               best_goal_node = goal_node
               print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))

      # 6. Visualization
      if display_map:
         rrt_dubins.draw_graph()

   else:
      if best_goal_node is None:
         print("Reached max iterations without finding a path")
         return None

   # 7. Extract path by tracing parent pointers from best goal to start
   rrt_dubins.node_list.append(best_goal_node)
   path = []
   node = best_goal_node
   while node is not None:
      path.append(node)
      node = node.parent
   path.reverse()
   return path
