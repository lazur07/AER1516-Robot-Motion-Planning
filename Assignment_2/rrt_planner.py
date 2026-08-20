"""
Assignment #2 Template file
"""

import random
import math
import numpy as np

"""
Problem Statement
--------------------
Implement the Rapidly-Exploring Random Trees (RRT) planning algorithm
for the problem setup provided by the RRT_dubins_problem class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file: rrt_planner.py. Your implementation
   can be tested by running dubins_path_problem.py (check the main function).
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
1. DO NOT rename the file rrt_planner.py for submission.
2. DO NOT change the rrt_planner function signature.
3. DO NOT import anything other than what is already imported in this file.
4. YOU MAY write additional helper functions in this file to reduce code repetition,
   but these functions can only be used inside the rrt_planner function
   (since only the rrt_planner function will be imported for grading).
"""


def rrt_planner(rrt_dubins, display_map=False):
   """
   Execute RRT planning using Dubins-style paths. Make sure to populate the node_list.

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
   x_min = rrt_dubins.x_lim[0]
   x_max = rrt_dubins.x_lim[1]
   y_min = rrt_dubins.y_lim[0]
   y_max = rrt_dubins.y_lim[1]

   # LOOP for max iterations
   for i in range(rrt_dubins.max_iter):

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

      # 2. Find the nearest existing node (Euclidean distance)
      nearest_node = min(rrt_dubins.node_list,
         key=lambda node: (node.x - random_state.x)**2 + (node.y - random_state.y)**2)

      # 2.1 Propagate a Dubins path from nearest node to random state
      new_node = rrt_dubins.propagate(nearest_node, random_state)

      # 3. Collision check, boundary check, and node insertion
      if rrt_dubins.check_collision(new_node) and is_path_within_bounds(new_node, x_min, x_max, y_min, y_max):
         rrt_dubins.node_list.append(new_node)

         # 3.1 Attempt to connect the new node to the goal
         goal_node = rrt_dubins.propagate(new_node, rrt_dubins.goal)
         if rrt_dubins.check_collision(goal_node) and is_path_within_bounds(goal_node, x_min, x_max, y_min, y_max):
            print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))
            rrt_dubins.node_list.append(goal_node)
            break

      # 4. Visualization
      if display_map:
         rrt_dubins.draw_graph()

   else:
      print("Reached max iterations without finding a path")
      return None

   # 5. Extract path by tracing parent pointers from goal to start
   path = []
   node = goal_node
   while node is not None:
      path.append(node)
      node = node.parent
   path.reverse()
   return path
