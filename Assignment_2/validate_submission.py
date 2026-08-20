"""
AER1516 Assignment 2 - Submission Validator
============================================
Use this script to check your RRT and RRT* implementations before submitting.

USAGE:
    Place this file in the same directory as your rrt_planner.py,
    rrt_star_planner.py, dubins_path_problem.py, and dubins_path_planning.py.

    Then run:
        python validate_submission.py

    This will run a subset of test cases and report PASS/FAIL for each.
    Passing all tests here does NOT guarantee full marks — the grading
    script uses additional test cases and checks.

NOTE: This script disables plotting. Do not modify this file.
"""

import io
import sys
import math
import random
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# Import student code and framework
# ---------------------------------------------------------------------------
try:
    from dubins_path_problem import RRT_dubins_problem
    from rrt_planner import rrt_planner
    from rrt_star_planner import rrt_star_planner
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure this file is in the same directory as your code files.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Validation helpers (simplified — grading uses more thorough checks)
# ---------------------------------------------------------------------------
GOAL_XY_TOL = 0.5
GOAL_YAW_TOL_DEG = 1.0


def _validate_path(rrt_problem, path):
    """Validate a planner's output. Returns (ok, message, cost)."""
    if path is None:
        return False, "Planner returned None", None
    if not isinstance(path, list) or len(path) < 2:
        return False, "Path must be a list with at least 2 nodes", None

    # Start check
    s = path[0]
    if abs(s.x - rrt_problem.start.x) > 0.01 or \
       abs(s.y - rrt_problem.start.y) > 0.01 or \
       abs(s.yaw - rrt_problem.start.yaw) > 0.01:
        return False, "First node does not match start state", None

    # Goal check
    e = path[-1]
    goal_dist = math.hypot(e.x - rrt_problem.goal.x, e.y - rrt_problem.goal.y)
    yaw_diff = abs(e.yaw - rrt_problem.goal.yaw)
    yaw_diff = min(yaw_diff, 2 * math.pi - yaw_diff)
    if goal_dist > GOAL_XY_TOL:
        return False, f"Last node too far from goal (dist={goal_dist:.3f})", None
    if yaw_diff > np.deg2rad(GOAL_YAW_TOL_DEG):
        return False, f"Last node yaw mismatch ({np.rad2deg(yaw_diff):.2f} deg)", None

    # Collision check
    for i, node in enumerate(path[1:], 1):
        if not rrt_problem.check_collision(node):
            return False, f"Collision at node {i} ({node.x:.2f}, {node.y:.2f})", None

    # Boundary check (node positions AND path waypoints)
    x_lo, x_hi = rrt_problem.x_lim
    y_lo, y_hi = rrt_problem.y_lim
    for i, node in enumerate(path):
        if node.x < x_lo or node.x > x_hi or node.y < y_lo or node.y > y_hi:
            return False, f"Node {i} position ({node.x:.2f}, {node.y:.2f}) outside map", None
        # Check all path waypoints (Dubins curves can extend beyond node positions)
        if hasattr(node, 'path_x') and node.path_x:
            for px, py in zip(node.path_x, node.path_y):
                if px < x_lo or px > x_hi or py < y_lo or py > y_hi:
                    return False, (
                        f"Node {i} path waypoint ({px:.2f}, {py:.2f}) outside map "
                        f"[{x_lo}, {x_hi}] x [{y_lo}, {y_hi}]"
                    ), None

    return True, "OK", path[-1].cost


def _run_planner(planner_fn, rrt_problem):
    """Run a planner, suppress its stdout, and catch exceptions."""
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = planner_fn(rrt_problem, display_map=False)
        finally:
            sys.stdout = old_stdout
        return result
    except Exception:
        sys.stdout = old_stdout if sys.stdout != old_stdout else sys.stdout
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Test cases (subset — grading uses more)
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": "A",
        "name": "No obstacles",
        "start": [1.0, 1.0, 0.0],
        "goal": [10.0, 10.0, 0.0],
        "obstacles": [],
        "map_area": [-2.0, 15.0, -2.0, 15.0],
        "max_iter": 300,
        "seed": 42,
    },
    {
        "id": "B",
        "name": "Single large obstacle",
        "start": [0.0, 5.0, 0.0],
        "goal": [10.0, 5.0, 0.0],
        "obstacles": [(5, 5, 3)],
        "map_area": [-2.0, 15.0, -2.0, 15.0],
        "max_iter": 500,
        "seed": 100,
    },
    {
        "id": "C",
        "name": "Perpendicular goal yaw",
        "start": [0.0, 0.0, 0.0],
        "goal": [10.0, 10.0, np.deg2rad(90.0)],
        "obstacles": [(5, 5, 2)],
        "map_area": [-2.0, 15.0, -2.0, 15.0],
        "max_iter": 500,
        "seed": 500,
    },
    {
        "id": "D",
        "name": "Default with obstacles",
        "start": [0.0, 0.0, np.deg2rad(-50.0)],
        "goal": [10.0, 10.0, np.deg2rad(50.0)],
        "obstacles": [
            (5, 5, 1), (3, 6, 2), (3, 8, 2),
            (3, 10, 2), (7, 5, 2), (9, 5, 2),
        ],
        "map_area": [-2.0, 15.0, -2.0, 15.0],
        "max_iter": 500,
        "seed": 50,
    },
]


def _make_problem(tc):
    """Create an RRT_dubins_problem from a test case dict."""
    return RRT_dubins_problem(
        start=tc["start"],
        goal=tc["goal"],
        obstacle_list=tc["obstacles"],
        map_area=tc["map_area"],
        max_iter=tc["max_iter"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("AER1516 Assignment 2 — Submission Validator")
    print("=" * 60)

    results = {"rrt": [], "rrt_star": []}

    # --- RRT tests ---
    print("\n--- RRT Planner Tests ---\n")
    for tc in TEST_CASES:
        random.seed(tc["seed"])
        np.random.seed(tc["seed"])
        prob = _make_problem(tc)
        path = _run_planner(rrt_planner, prob)
        ok, msg, cost = _validate_path(prob, path)
        status = "PASS" if ok else "FAIL"
        cost_str = f"  (cost={cost:.2f})" if cost is not None else ""
        print(f"  [{status}] Test {tc['id']}: {tc['name']}{cost_str}")
        if not ok:
            print(f"         Reason: {msg}")
        results["rrt"].append(ok)

    # --- RRT* tests ---
    print("\n--- RRT* Planner Tests ---\n")
    for tc in TEST_CASES:
        random.seed(tc["seed"])
        np.random.seed(tc["seed"])
        prob = _make_problem(tc)
        path = _run_planner(rrt_star_planner, prob)
        ok, msg, cost = _validate_path(prob, path)
        status = "PASS" if ok else "FAIL"
        cost_str = f"  (cost={cost:.2f})" if cost is not None else ""
        print(f"  [{status}] Test {tc['id']}: {tc['name']}{cost_str}")
        if not ok:
            print(f"         Reason: {msg}")
        results["rrt_star"].append(ok)

    # --- RRT* vs RRT cost comparison ---
    print("\n--- RRT* vs RRT Cost Comparison ---\n")
    comp_config = {
        "start": [0.0, 0.0, np.deg2rad(-50.0)],
        "goal": [10.0, 10.0, np.deg2rad(50.0)],
        "obstacles": [
            (5, 5, 1), (3, 6, 2), (3, 8, 2),
            (3, 10, 2), (7, 5, 2), (9, 5, 2),
        ],
        "map_area": [-2.0, 15.0, -2.0, 15.0],
        "max_iter": 500,
    }
    comp_seeds = [10, 30]
    rrt_costs = []
    rrtstar_costs = []

    for seed in comp_seeds:
        # RRT run
        random.seed(seed)
        np.random.seed(seed)
        prob = RRT_dubins_problem(
            start=comp_config["start"],
            goal=comp_config["goal"],
            obstacle_list=comp_config["obstacles"],
            map_area=comp_config["map_area"],
            max_iter=comp_config["max_iter"],
        )
        path = _run_planner(rrt_planner, prob)
        ok, _, cost = _validate_path(prob, path)
        if ok:
            rrt_costs.append(cost)

        # RRT* run
        random.seed(seed)
        np.random.seed(seed)
        prob = RRT_dubins_problem(
            start=comp_config["start"],
            goal=comp_config["goal"],
            obstacle_list=comp_config["obstacles"],
            map_area=comp_config["map_area"],
            max_iter=comp_config["max_iter"],
        )
        path = _run_planner(rrt_star_planner, prob)
        ok, _, cost = _validate_path(prob, path)
        if ok:
            rrtstar_costs.append(cost)

    if len(rrt_costs) == len(comp_seeds) and len(rrtstar_costs) == len(comp_seeds):
        avg_rrt = sum(rrt_costs) / len(rrt_costs)
        avg_rrtstar = sum(rrtstar_costs) / len(rrtstar_costs)
        print(f"  RRT  avg cost: {avg_rrt:.2f}")
        print(f"  RRT* avg cost: {avg_rrtstar:.2f}")
        if avg_rrtstar < avg_rrt:
            print("  [PASS] RRT* produces lower average cost than RRT")
            cost_comp_ok = True
        else:
            print("  [FAIL] RRT* should produce lower average cost than RRT")
            cost_comp_ok = False
    else:
        print(f"  [FAIL] Not all runs produced valid paths "
              f"(RRT: {len(rrt_costs)}/{len(comp_seeds)}, "
              f"RRT*: {len(rrtstar_costs)}/{len(comp_seeds)})")
        cost_comp_ok = False

    # --- Summary ---
    rrt_pass = sum(results["rrt"])
    rrt_total = len(results["rrt"])
    star_pass = sum(results["rrt_star"])
    star_total = len(results["rrt_star"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  RRT:            {rrt_pass}/{rrt_total} passed")
    print(f"  RRT*:           {star_pass}/{star_total} passed")
    print(f"  Cost comparison: {'PASS' if cost_comp_ok else 'FAIL'}")

    all_pass = (rrt_pass == rrt_total and star_pass == star_total and cost_comp_ok)
    if all_pass:
        print("\n  All checks passed! Your code looks ready to submit.")
        print("  Note: The grading script uses additional test cases.")
    else:
        print("\n  Some checks failed. Fix the issues above before submitting.")

    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
