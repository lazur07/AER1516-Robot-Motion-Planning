"""
Stress test: run RRT and RRT* for N random seeds, then generate:
  1. figs/environment.png     -- the obstacle environment used
  2. figs/bar_chart.png       -- bar chart comparing average path lengths
  3. figs/best_save_case.png  -- one seed with largest RRT − RRT* path-length save

Usage:
    python stress_test.py [N] [MAX_ITER]
    e.g.  python stress_test.py 1000 10000
"""
import random
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from dubins_path_problem import RRT_dubins_problem, check_path, get_path

# Golden ratio for figure aspect: width / height = (1 + sqrt(5)) / 2
PHI = (1.0 + math.sqrt(5.0)) / 2.0

# Times New Roman for all text in figures
plt.rcParams["font.family"] = "Times New Roman"

# ── Configuration ──────────────────────────────────────────────────
N        = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MAX_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

OBSTACLE_LIST = [
    (2.8, 8.3, 1.10),
    (5.1, 8.2, 0.86),
    (6.0, 6.1, 0.78),
    (4.1, 5.2, 0.82),
    (7.8, 5.1, 0.90),
    (9.6, 4.6, 0.62),
    (11.0, 5.2, 0.50),
]
START = [0.0, 0.0, np.deg2rad(-50.0)]
GOAL  = [10.0, 10.0, np.deg2rad(50.0)]
MAP   = [-2.0, 15.0, -2.0, 15.0]

# ── Color palette (matches comparison.tex) ─────────────────────────
PONE  = (30 / 255, 55 / 255, 101 / 255)   # #1E3765
PTWO  = (143 / 255, 129 / 255, 116 / 255) # #8F8174
QUOTE = (157 / 255, 187 / 255, 216 / 255) # #9DBBD8
NOTE  = (111 / 255, 199 / 255, 234 / 255) # #6FC7EA

os.makedirs("figs", exist_ok=True)


# ── 1. Environment plot ───────────────────────────────────────────
def plot_environment():
    fig, ax = plt.subplots(figsize=(5, 5))

    for ox, oy, r in OBSTACLE_LIST:
        circle = Circle((ox, oy), r, facecolor=QUOTE, edgecolor=PONE,
                         linewidth=1, alpha=0.85)
        ax.add_patch(circle)

    ax.annotate("", xy=(START[0] + 1.0 * math.cos(START[2]),
                         START[1] + 1.0 * math.sin(START[2])),
                xytext=(START[0], START[1]),
                arrowprops=dict(arrowstyle="-|>", color=PONE, lw=1.2, alpha=0.7))
    ax.plot(START[0], START[1], "o", color=PONE, ms=7, zorder=5)
    ax.text(START[0] + 0.5, START[1] - 1, "Start",
            fontsize=12, color=PONE, fontweight="bold")

    ax.annotate("", xy=(GOAL[0] + 1.0 * math.cos(GOAL[2]),
                         GOAL[1] + 1.0 * math.sin(GOAL[2])),
                xytext=(GOAL[0], GOAL[1]),
                arrowprops=dict(arrowstyle="-|>", color=PTWO, lw=1.2, alpha=0.7))
    ax.plot(GOAL[0], GOAL[1], "s", color=PTWO, ms=7, zorder=5)
    ax.text(GOAL[0] + 0.5, GOAL[1] - 1, "Goal",
            fontsize=12, color=PTWO, fontweight="bold")

    ax.set_xlim(MAP[0] - 0.5, MAP[1] + 0.5)
    ax.set_ylim(MAP[2] - 0.5, MAP[3] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Obstacle Environment", fontsize=11, fontweight="bold",
                 color=PONE)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig("figs/environment.png", dpi=240)
    plt.close(fig)
    print("Saved figs/environment.png")


# ── 2. Run trials ─────────────────────────────────────────────────
def run_trial(planner_name, seed):
    random.seed(seed)
    prob = RRT_dubins_problem(
        start=START, goal=GOAL,
        obstacle_list=OBSTACLE_LIST,
        map_area=MAP, max_iter=MAX_ITER,
    )
    if planner_name == "rrt":
        nodes = prob.rrt_planning(display_map=False)
    else:
        nodes = prob.rrt_star_planning(display_map=False)

    if nodes is None:
        return None, False

    valid = check_path(prob, nodes)
    cost = nodes[-1].cost if nodes else None
    return cost, valid


def run_trial_return_path(planner_name, seed):
    """Run one trial and return (cost, valid, path_node_list) for plotting."""
    random.seed(seed)
    prob = RRT_dubins_problem(
        start=START, goal=GOAL,
        obstacle_list=OBSTACLE_LIST,
        map_area=MAP, max_iter=MAX_ITER,
    )
    if planner_name == "rrt":
        nodes = prob.rrt_planning(display_map=False)
    else:
        nodes = prob.rrt_star_planning(display_map=False)

    if nodes is None:
        return None, False, None
    valid = check_path(prob, nodes)
    cost = nodes[-1].cost if nodes else None
    return cost, valid, nodes


def run_all_trials():
    results = {}
    for planner in ["rrt", "rrt_star"]:
        successes, failures, invalid = 0, 0, 0
        costs = []
        cost_by_seed = [None] * N  # cost_by_seed[seed] = cost if success else None
        t0 = time.time()

        for seed in range(N):
            if seed % 100 == 0:
                print(f"  [{planner:>8}] seed {seed}/{N} ...")
            cost, valid = run_trial(planner, seed)
            if cost is None:
                failures += 1
            elif not valid:
                invalid += 1
                print(f"  [{planner}] seed={seed} INVALID PATH")
            else:
                successes += 1
                costs.append(cost)
                cost_by_seed[seed] = cost

        elapsed = time.time() - t0
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        results[planner] = {
            "successes": successes,
            "failures": failures,
            "invalid": invalid,
            "costs": costs,
            "cost_by_seed": cost_by_seed,
            "avg": avg_cost,
            "min": min(costs) if costs else 0.0,
            "max": max(costs) if costs else 0.0,
            "time": elapsed,
        }

        print(f"\n{'=' * 50}")
        print(f"Planner : {planner.upper()}")
        print(f"Trials  : {N}  |  max_iter={MAX_ITER}")
        print(f"Success : {successes}/{N} ({100 * successes / N:.1f}%)")
        print(f"Failed  : {failures}  |  Invalid: {invalid}")
        print(f"Cost    : avg={avg_cost:.3f}  "
              f"min={results[planner]['min']:.3f}  "
              f"max={results[planner]['max']:.3f}")
        print(f"Time    : {elapsed:.1f}s")

    return results


def find_best_save_seed(results):
    """Return (seed, rrt_cost, rrt_star_cost, save) with largest RRT − RRT* save."""
    best_seed = None
    best_save = -1.0
    rrt_c = None
    rrt_star_c = None
    for seed in range(N):
        c_rrt = results["rrt"]["cost_by_seed"][seed]
        c_star = results["rrt_star"]["cost_by_seed"][seed]
        if c_rrt is not None and c_star is not None:
            save = c_rrt - c_star
            if save > best_save:
                best_save = save
                best_seed = seed
                rrt_c = c_rrt
                rrt_star_c = c_star
    return best_seed, rrt_c, rrt_star_c, best_save


def plot_best_save_case(best_seed, rrt_cost, rrt_star_cost, save):
    """Plot environment and both paths for the seed that had the biggest RRT→RRT* save."""
    _, _, rrt_nodes = run_trial_return_path("rrt", best_seed)
    _, _, rrt_star_nodes = run_trial_return_path("rrt_star", best_seed)
    if rrt_nodes is None or rrt_star_nodes is None:
        print("  Skipping best-save plot (missing path).")
        return

    path_rrt = get_path(rrt_nodes)
    path_star = get_path(rrt_star_nodes)
    x_rrt = [p[0] for p in path_rrt]
    y_rrt = [p[1] for p in path_rrt]
    x_star = [p[0] for p in path_star]
    y_star = [p[1] for p in path_star]

    fig, ax = plt.subplots(figsize=(5, 5))

    for ox, oy, r in OBSTACLE_LIST:
        circle = Circle((ox, oy), r, facecolor=QUOTE, edgecolor=PONE,
                         linewidth=1, alpha=0.85)
        ax.add_patch(circle)

    ax.plot(x_rrt, y_rrt, "-", color=PTWO, linewidth=1, label=f"RRT ({rrt_cost:.2f})", zorder=2)
    ax.plot(x_star, y_star, "-", color=PONE, linewidth=1, label=f"RRT* ({rrt_star_cost:.2f})", zorder=2)

    ax.annotate("", xy=(START[0] + 1.0 * math.cos(START[2]), START[1] + 1.0 * math.sin(START[2])),
                xytext=(START[0], START[1]),
                arrowprops=dict(arrowstyle="-|>", color=PONE, lw=1.2, alpha=0.7))
    ax.plot(START[0], START[1], "o", color=PONE, ms=6, zorder=5)
    ax.annotate("", xy=(GOAL[0] + 1.0 * math.cos(GOAL[2]), GOAL[1] + 1.0 * math.sin(GOAL[2])),
                xytext=(GOAL[0], GOAL[1]),
                arrowprops=dict(arrowstyle="-|>", color=PTWO, lw=1.2, alpha=0.7))
    ax.plot(GOAL[0], GOAL[1], "s", color=PTWO, ms=6, zorder=5)

    ax.set_xlim(MAP[0] - 0.5, MAP[1] + 0.5)
    ax.set_ylim(MAP[2] - 0.5, MAP[3] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Seed {best_seed}: largest path-length save (RRT - RRT* = {save:.2f})",
                 fontsize=11, fontweight="bold", color=PONE)
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig("figs/best_save_case.png", dpi=240)
    plt.close(fig)
    print("Saved figs/best_save_case.png")


# ── 3. Bar chart ──────────────────────────────────────────────────
def plot_bar_chart(results):
    labels = ["RRT", "RRT*"]
    avgs   = [results["rrt"]["avg"], results["rrt_star"]["avg"]]
    colors = [PTWO, PONE]

    # figsize: golden ratio (width, width/phi)
    fig_w = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / PHI))
    bars = ax.bar(labels, avgs, width=0.45, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)

    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=PONE)

    ax.set_ylabel("Average Path Length", fontsize=10)
    ax.set_title(f"RRT vs. RRT* ({N} Trials, max_iter={MAX_ITER})",
                 fontsize=11, fontweight="bold", color=PONE)
    ax.set_ylim(0, max(avgs) * 1.18)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("figs/bar_chart.png", dpi=240)
    plt.close(fig)
    print("Saved figs/bar_chart.png")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_environment()
    results = run_all_trials()
    plot_bar_chart(results)

    best_seed, rrt_cost, rrt_star_cost, save = find_best_save_seed(results)
    if best_seed is not None:
        print(f"\nBest save: seed={best_seed}  RRT={rrt_cost:.3f}  RRT*={rrt_star_cost:.3f}  save={save:.3f}")
        plot_best_save_case(best_seed, rrt_cost, rrt_star_cost, save)
    else:
        print("\nNo seed with both RRT and RRT* success; skipping best-save plot.")

    print("\nDone. Seed range: 0 .. %d" % (N - 1))
