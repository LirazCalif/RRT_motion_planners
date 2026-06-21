import numpy as np
import os
import json
from datetime import datetime
from twoD.environment import MapEnvironment
from twoD.dot_environment import MapDotEnvironment
from twoD.dot_building_blocks import DotBuildingBlocks2D
from twoD.building_blocks import BuildingBlocks2D
from twoD.dot_visualizer import DotVisualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR
from AStarPlanner import AStarPlanner
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
from RRTStarPlanner import RRTStarPlanner
from twoD.visualizer import Visualizer

MAP_DETAILS = {"json_file": "twoD/map1.json", "start": np.array([10,10]), "goal": np.array([4, 6])}
# MAP_DETAILS = {"json_file": "twoD/map2.json", "start": np.array([360, 150]), "goal": np.array([100, 200])}


def run_dot_2d_astar():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = AStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])
    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, expanded_nodes=planner.expanded_nodes, show_map=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_dot_2d_rrt():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    t=0
    cost=0
    for ext_mode in ["E1","E2"]:
        planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=0.05)
        # execute plan
        plan = planner.plan()
        t+=planner.t
        cost+=planner.path_cost
        DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

    # print(f"average cost: {cost/10}, average time: {t/10}")
    # DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_dot_2d_rrt_star():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.2, k=None, max_step_size=0.3)
    planner.timeout=10

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

import random
def run_2d_rrt_star_motion_planning():
    MAP_DETAILS = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal": np.array([0.3, 0.15, 1.0, 1.1]),
    }

    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)

    configs = {
        0.05: 5 * 112,  # 560 sec
        0.2: 5 * 10.76  # ~53.8 sec
    }

    runs = 10
    results = []

    for bias, timeout in configs.items():
        all_costs = []
        for i in range(runs):
            seed=i
            np.random.seed(seed)
            random.seed(seed)

            planner = RRTStarPlanner(
                bb=bb,
                start=MAP_DETAILS["start"],
                goal=MAP_DETAILS["goal"],
                ext_mode="E2",
                goal_prob=bias,
                max_step_size=0.3,
                max_itr=None,
                stop_on_goal=False
            )
            planner.two_d=True

            planner.timeout = timeout
            plan = planner.plan()
            if plan is None:
                path_cost = np.inf
                time=planner.timeout
            else:
                path_cost = planner.path_cost
                time=float(planner.t)
                all_costs.append(planner.best_cost_per_t)
                print(f"RRT*: bias={bias}, run={i}, final_cost={path_cost:.2f}")

            results.append({
                "bias": bias,
                "ext_mode": "E2",
                "run_index": i + 1,
                "seed": seed,
                "path_cost": float(path_cost),
                "time": time,
                "best_cost":all_costs,
                "path": plan.tolist() if plan is not None else None
            })


    with open("rrt_star_2d_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved RRT* 2D results to rrt_star_2d_results.json")





def run_2d_rrt_motion_planning():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)

    results = []


    for bias in [0.05,0.2]:
        for ext_mode in ["E2"]:
            i = 0
            while i < 10:
                planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=bias)
                # execute plan
                plan = planner.plan()
                # Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

                if plan is not None:
                    results.append({
                        "bias": bias,
                        "ext_mode": ext_mode,
                        "run_index": i + 1,
                        "path_cost": float(planner.path_cost),
                        "time": float(planner.t),
                        "path": plan.tolist()
                    })
                    i += 1
            print(f"for {ext_mode}")
            print(f"cost: {planner.path_cost}")
            print(f"time: {planner.t}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rrt_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

    return filename

import json
import numpy as np
import matplotlib.pyplot as plt

def rrt_2d_results(json_file: str):
    # Load results
    with open(json_file, "r") as f:
        results = json.load(f)

    # Organize results by ext_mode and bias
    data = {}
    for r in results:
        mode = r["ext_mode"]
        bias = r["bias"]
        key = (mode, bias)
        if key not in data:
            data[key] = {"time": [], "cost": [], "paths": []}  # store paths too
        data[key]["time"].append(float(r["time"]))
        data[key]["cost"].append(float(r["path_cost"]))
        # save path if available, else None
        data[key]["paths"].append(r.get("path", None))

    # Compute mean and std
    summary = {}
    print("Summary Table:")
    for (mode, bias), vals in data.items():
        times = np.array(vals["time"])
        costs = np.array(vals["cost"])
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        summary[(mode, bias)] = {
            "mean_time": mean_time,
            "std_time": std_time,
            "mean_cost": mean_cost,
            "std_cost": std_cost
        }

        print(f"{mode}, Bias={int(bias*100)}%")
        print(f"\tTime:  Mean: {mean_time:.3f}, Stdev: {std_time:.3f}")
        print(f"\tCost:  Mean: {mean_cost:.3f}, Stdev: {std_cost:.3f}")

    # Plot Cost vs Time
    plt.figure(figsize=(8, 6))
    colors = {0.05: "blue", 0.2: "red"}

    for (mode, bias), vals in data.items():
        # Sort by time to make a proper line
        times = np.array(vals["time"])
        costs = np.array(vals["cost"])
        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        costs_sorted = costs[sort_idx]

        plt.scatter(
            times_sorted,
            costs_sorted,
            c=colors[bias],
            label=f"{mode}, bias={int(bias * 100)}%",
            alpha=0.7
        )
    plt.xlabel("Time (sec)")
    plt.ylabel("Path Cost")
    plt.title("Cost vs Time For RRT")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Pick representative runs (closest to mean cost)
    representative_runs = []
    for (mode, bias), vals in data.items():
        costs = np.array(vals["cost"])
        times = np.array(vals["time"])
        paths = vals["paths"]
        mean_cost = np.mean(costs)
        idx = np.argmin(np.abs(costs - mean_cost))

        rep_run = {
            "ext_mode": mode,
            "bias": bias,
            "time": float(times[idx]),
            "cost": float(costs[idx]),
            "path": paths[idx]
        }
        representative_runs.append(rep_run)

    print("\nRepresentative runs for visualization:")
    for run in representative_runs:
        # Print nicely with 2 decimals
        print(f"{run['ext_mode']}, Bias={int(run['bias'] * 100)}%, "
              f"Time={run['time']:.2f}, Cost={run['cost']:.2f}")

        # Setup environment and BB for visualization
        MAP_DETAILS = {"json_file": "twoD/map_mp.json",
                       "start": np.array([0.78, -0.78, 0.0, 0.0]),
                       "goal": np.array([0.3, 0.15, 1.0, 1.1])}
        planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
        bb = BuildingBlocks2D(planning_env)

        # Visualize the path if available
        # if run["path"] is not None:
        #     Visualizer(bb).visualize_plan(plan=np.array(run["path"]),
        #                                   start=MAP_DETAILS["start"],
        #                                   goal=MAP_DETAILS["goal"])
        # else:
        #     print("No path available for this run.")

    return data, summary, representative_runs

def run_2d_rrt_inspection_planning():
    MAP_DETAILS = {
        "json_file": "twoD/map_ip.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0])
    }
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)

    coverages = [0.75]
    runs_per_cov = 10

    results = []

    for coverage in coverages:
        for run_id in range(runs_per_cov):
            planner = RRTInspectionPlanner(
                bb=bb,
                start=MAP_DETAILS["start"],
                ext_mode="E2",
                goal_prob=0.2,
                coverage=coverage
            )

            plan = planner.plan()

            if plan is not None:
                results.append({
                    "coverage": coverage,
                    "run_index": run_id,
                    "time": float(planner.t),
                    "cost": float(planner.path_cost),
                    "plan": plan.tolist()
                })

            print(f"Coverage={coverage}, Run={run_id}, "
                  f"Time={planner.t:.2f}, Cost={planner.path_cost:.2f}")

    # save all results
    with open("rrt_inspection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved inspection results to rrt_inspection_results.json")
    # Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def rrt_2d_inspection_results(json_file):
    import json
    import numpy as np

    with open(json_file, "r") as f:
        results = json.load(f)

    coverages = sorted(set(r["coverage"] for r in results))

    print("\nInspection Planning Results:")
    print("--------------------------------")

    representative_runs = []

    for cov in coverages:
        runs = [r for r in results if r["coverage"] == cov]

        times = np.array([r["time"] for r in runs])
        costs = np.array([r["cost"] for r in runs])

        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)

        print(f"Coverage = {cov}")
        print(f"\tTime  Mean: {mean_time:.2f}, Std: {std_time:.2f}")
        print(f"\tCost  Mean: {mean_cost:.2f}, Std: {std_cost:.2f}")

        # representative run = closest to mean cost
        idx = np.argmin(np.abs(costs - mean_cost))
        representative_runs.append(runs[idx])

    # visualize representative runs
    MAP_DETAILS = {
        "json_file": "twoD/map_ip.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0])
    }
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)

    print("\nVisualizing representative runs:")
    for r in representative_runs:
        print(f"Coverage={r['coverage']}, "
              f"Time={r['time']:.2f}, Cost={r['cost']:.2f}")

        Visualizer(bb).visualize_plan(
            plan=np.array(r["plan"]),
            start=MAP_DETAILS["start"]
        )

def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1 )

    wall_config_val=[130,-70,90,-90,-90,0]
    wall_val=bb.config_validity_checker(wall_config_val)
    print("Checking configure validity, when it hit the wall: the function return",wall_val )

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------


    rrt_star_planner = RRTStarPlanner(max_step_size=0.5,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr=4000,
                                      stop_on_goal=True,
                                      bb=bb,
                                      goal_prob=0.05,
                                      ext_mode="E2")

    path = rrt_star_planner.plan()

    if path is not None:

        # create a folder for the experiment
        # Format the time string as desired (YYYY-MM-DD_HH-MM-SS)
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # create the folder
        exps_folder_name = os.path.join(os.getcwd(), "exps")
        if not os.path.exists(exps_folder_name):
            os.mkdir(exps_folder_name)
        exp_folder_name = os.path.join(exps_folder_name, "exp_p_bias_"+ str(rrt_star_planner.goal_prob) + "_max_step_size_" + str(rrt_star_planner.max_step_size) + "_" + time_str)
        if not os.path.exists(exp_folder_name):
            os.mkdir(exp_folder_name)

        # save the path
        np.save(os.path.join(exp_folder_name, 'path'), path)

        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost(path)))

        visualizer.show_path(path)

        max_step_sizes = [0.05, 0.075, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4]
        p_biases = [0.05, 0.2]
        runs = 20

        exps_folder = os.path.join(os.getcwd(), "rrt_star_exps")
        os.makedirs(exps_folder, exist_ok=True)

        for p_bias in p_biases:
            avg_results = {}
            for max_step in max_step_sizes:
                final_paths = []
                all_costs = []
                all_success = []

                for run in range(runs):
                    rrt_star_planner = RRTStarPlanner(
                        max_step_size=max_step,
                        start=env2_start,
                        goal=env2_goal,
                        max_itr=2000,
                        stop_on_goal=False,
                        bb=bb,
                        goal_prob=p_bias,
                        ext_mode="E2",
                    )
                    rrt_star_planner.three_d=True

                    path = rrt_star_planner.plan()
                    final_paths.append(path.tolist() if path is not None else None)
                    all_costs.append(rrt_star_planner.best_cost_per_iter.tolist())
                    all_success.append(rrt_star_planner.success_per_iter.tolist())

                # Save raw results
                raw_data = {
                    "p_bias": p_bias,
                    "max_step": max_step,
                    "paths": final_paths,
                    "cost_per_iter": all_costs,
                    "success_per_iter": all_success
                }

                raw_file = os.path.join(exps_folder, f"raw_p_bias_{p_bias}_maxstep_{max_step}.json")
                with open(raw_file, "w") as f:
                    json.dump(raw_data, f, indent=2)

                # Compute averages over runs
                all_costs_np = np.array(all_costs)
                all_success_np = np.array(all_success)

                avg_cost = []
                for i in range(all_costs_np.shape[1]):
                    mask = all_costs_np[:, i] < np.inf
                    avg_cost.append(float(np.mean(all_costs_np[mask, i])) if np.any(mask) else None)

                avg_success = [float(x) for x in np.mean(all_success_np, axis=0)]

                # Store average results
                avg_results[max_step] = {
                    "avg_cost": avg_cost,
                    "avg_success": avg_success
                }

            # Save average results for p_bias
            avg_file = os.path.join(exps_folder, f"avg_p_bias_{p_bias}.json")
            with open(avg_file, "w") as f:
                json.dump(avg_results, f, indent=2)




import matplotlib.pyplot as plt
def plot_results_run_3d():
    RESULTS_DIR = "rrt_star_exps_results"
    p_bias =[0.05, 0.2]
    for p in p_bias:
        plot_results_3d_per_bias(p, RESULTS_DIR)


def plot_results_3d_per_bias(p_bias,RESULTS_DIR):
    avg_file = os.path.join(RESULTS_DIR, f"avg_p_bias_{p_bias}.json")

    with open(avg_file, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(14, 6))

    # Cost vs Iteration
    plt.subplot(1, 2, 1)
    for max_step, results in data.items():
        cost = np.array(results["avg_cost"], dtype=float)
        iterations = np.arange(len(cost))

        # Mask undefined values (before first solution)
        valid = ~np.isnan(cost)
        plt.plot(iterations[valid], cost[valid], label=f"step={max_step}")

    plt.xlabel("Iteration")
    plt.ylabel("Path Cost")
    plt.title(f"Cost vs Iteration (p_bias={p_bias})")
    plt.legend()
    plt.grid(True)

    #  Success Rate vs Iteration
    plt.subplot(1, 2, 2)
    for max_step, results in data.items():
        success = np.array(results["avg_success"])
        iterations = np.arange(len(success))

        plt.plot(iterations, success, label=f"step={max_step}")

    plt.xlabel("Iteration")
    plt.ylabel("Success Rate")
    plt.title(f"Success Rate vs Iteration (p_bias={p_bias})")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def find_best_path_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(
        transform=transform,
        ur_params=ur_params,
        env=env,
        resolution=0.1
    )

    # Dummy planner ONLY for cost computation
    rrt_star_planner = RRTStarPlanner(
        max_step_size=0.1,  # irrelevant here
        start=None,
        goal=None,
        max_itr=1,
        stop_on_goal=False,
        bb=bb,
        goal_prob=0.0,
        ext_mode="E2"
    )

    #  Search best path
    best_cost = np.inf
    best_path = None
    best_meta = None

    RESULTS_DIR = "rrt_star_exps_results"

    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith("raw_p_bias"):
            continue

        with open(os.path.join(RESULTS_DIR, fname), "r") as f:
            data = json.load(f)

        p_bias = data["p_bias"]
        max_step = data["max_step"]
        paths = data["paths"]

        for run_id, path in enumerate(paths):
            if path is None:
                continue

            path_np = np.array(path)
            cost = rrt_star_planner.compute_cost(path_np)

            if cost < best_cost:
                best_cost = cost
                best_path = path_np
                best_meta = (p_bias, max_step, run_id)

    # Safety check
    if best_path is None:
        print("No valid path found in any experiment.")
        return

    #  Report
    print("Best path found:")
    print(f"  Cost      : {best_cost:.4f}")
    print(f"  p_bias    : {best_meta[0]}")
    print(f"  max_step  : {best_meta[1]}")
    print(f"  run index : {best_meta[2]}")
    print(f"  waypoints : {len(best_path)}")
    print(f" best path in degrees : {np.round(np.rad2deg(best_path),2)}")

    # Save best path in JSON
    save_dir = os.path.join(os.getcwd(), "best_paths")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "best_path_3d.json")

    best_path_data = {
        "best_cost": float(best_cost),
        "p_bias": best_meta[0],
        "max_step": best_meta[1],
        "run_index": best_meta[2],
        "waypoints": len(best_path),
        "path": best_path.tolist()  # convert numpy array to list for JSON
    }

    with open(save_file, "w") as f:
        json.dump(best_path_data, f, indent=2)

    print(f"Best path saved to {save_file}")

    #  Visualize
    visualizer = Visualize_UR(
        ur_params=ur_params,
        env=env,
        transform=transform,
        bb=bb
    )
    visualizer.show_path(best_path)


#
# def plot_2d_rrt_star_results(json_file="rrt_star_2d_results1.json"):
#     # Load results
#     with open(json_file, "r") as f:
#         results = json.load(f)
#
#     p_biases = sorted(set(r["bias"] for r in results))
#     bias_colors = {0.05: "blue", 0.2: "red"}
#     all_final_costs = {}
#
#     for p_bias in p_biases:
#         runs = [r for r in results if r["bias"] == p_bias]
#         if not runs:
#             continue
#
#         plt.figure(figsize=(8, 6))
#         final_costs = []
#
#         # Determine max timeout across all runs
#         max_timeout = 0
#         for run in runs[:10]:
#             best_cost = run.get("best_cost", [])
#             flat = [pair for sublist in best_cost for pair in sublist if len(pair) == 2]
#             if flat:
#                 times, costs = zip(*flat)
#                 max_timeout = max(max_timeout, max(times))
#
#         # Plot each run
#         for run in runs[:10]:
#             best_cost = run.get("best_cost", [])
#             if not best_cost:
#                 continue
#
#             # Flatten nested best_cost
#             flat = []
#             for sublist in best_cost:
#                 for pair in sublist:
#                     if len(pair) == 2:
#                         flat.append(pair)
#             if not flat:
#                 continue
#
#             # Sort by time
#             flat.sort(key=lambda x: x[0])
#             times, costs = zip(*flat)
#             times = list(times)
#             costs = list(costs)
#
#             # Carry-forward last finite cost
#             for i in range(len(costs)):
#                 if not np.isfinite(costs[i]):
#                     costs[i] = costs[i-1] if i > 0 else 0
#
#             # Extend to max timeout
#             if times[-1] < max_timeout:
#                 times.append(max_timeout)
#                 costs.append(costs[-1])
#
#             plt.plot(times, costs, label=f"Run {run['run_index']}", alpha=0.7)
#
#             final_costs.append(costs[-1])
#
#         plt.xlabel("Time (sec)")
#         plt.ylabel("Solution Cost")
#         plt.title(f"RRT* 2D – Cost vs Time (bias={int(p_bias*100)}%)")
#         plt.grid(True)
#         plt.legend()
#         plt.show()
#
#         all_final_costs[p_bias] = final_costs
#
#     # --- Final Cost per Run (scatter plot) ---
#     plt.figure(figsize=(8, 5))
#     for i, p_bias in enumerate(p_biases):
#         costs = all_final_costs.get(p_bias, [])
#         x = np.arange(len(costs))
#         plt.scatter(x + i*0.05, costs, label=f"bias={int(p_bias*100)}%")  # small offset
#
#     plt.xlabel("Run Index")
#     plt.ylabel("Final Cost")
#     plt.title("RRT* 2D – Final Cost per Run")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#     # --- Representative run visualization ---
#     MAP_DETAILS = {
#         "json_file": "twoD/map_mp.json",
#         "start": np.array([0.78, -0.78, 0.0, 0.0]),
#         "goal": np.array([0.3, 0.15, 1.0, 1.1]),
#     }
#     planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
#     bb = BuildingBlocks2D(planning_env)
#
#     for p_bias in p_biases:
#         runs = [r for r in results if r.get("bias") == p_bias]
#         if not runs:
#             continue
#
#         final_costs = [r.get("path_cost", np.inf) for r in runs if np.isfinite(r.get("path_cost", np.inf))]
#         if not final_costs:
#             continue
#
#         mean_cost = np.mean(final_costs)
#         idx = np.argmin(np.abs(np.array([r.get("path_cost", np.inf) for r in runs]) - mean_cost))
#         rep_run = runs[idx]
#
#         print(f"Representative run for bias={p_bias}: seed={rep_run.get('seed','?')}, final cost={rep_run.get('path_cost',0):.2f}")
#
#         if rep_run.get("path"):
#             Visualizer(bb).visualize_plan(
#                 plan=np.array(rep_run["path"]),
#                 start=MAP_DETAILS["start"],
#                 goal=MAP_DETAILS["goal"]
#             )
#         else:
#             print("[INFO] No path for representative run, visualization skipped.")

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_rrt_star_results(json_file="rrt_star_2d_results.json"):
    # Load results
    with open(json_file, "r") as f:
        results = json.load(f)

    p_biases = sorted(set(r["bias"] for r in results))
    bias_colors = {0.05: "blue", 0.2: "red"}
    all_final_costs = {}

    # 1. Convergence Plot: Cost vs Time
    for p_bias in p_biases:
        runs = [r for r in results if r["bias"] == p_bias]
        if not runs:
            continue

        plt.figure(figsize=(10, 6))
        final_costs_for_stats = []

        for run in runs:
            best_cost_data = run.get("best_cost", [])
            time_cost_pairs = []
            for batch in best_cost_data:
                for pair in batch:
                    if len(pair) == 2:
                        time_cost_pairs.append(pair)

            if not time_cost_pairs:
                continue

            time_cost_pairs.sort(key=lambda x: x[0])
            times, costs = zip(*time_cost_pairs)
            plt.plot(times, costs, label=f"Run {run['run_index']}", alpha=0.6, linewidth=1.2)

            cost = run.get("path_cost", np.nan)
            if np.isfinite(cost):
                final_costs_for_stats.append(cost)

        plt.xlabel("Time (seconds)", fontsize=10)
        plt.ylabel("Solution Cost", fontsize=10)
        plt.title(f"RRT* 2D Convergence (Goal Bias = {int(p_bias * 100)}%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()

        all_final_costs[p_bias] = final_costs_for_stats

        if final_costs_for_stats:
            print(f"\nBias = {p_bias} Statistics:")
            print(f"  Mean final cost: {np.mean(final_costs_for_stats):.4f}")
            print(f"  Min/Max cost:    {np.min(final_costs_for_stats):.4f} / {np.max(final_costs_for_stats):.4f}")

    # 2. Final Cost Scatter Plot
    if all_final_costs:
        plt.figure(figsize=(10, 6))
        for p_bias in p_biases:
            costs = all_final_costs.get(p_bias, [])
            if costs:
                x = np.arange(1, len(costs) + 1)
                plt.scatter(x, costs, label=f"Bias = {int(p_bias * 100)}%",
                            color=bias_colors.get(p_bias, 'black'), s=60, alpha=0.7)
                plt.axhline(y=np.mean(costs), color=bias_colors.get(p_bias, 'black'),
                            linestyle='--', alpha=0.4, label=f"Mean (Bias {p_bias})")

        plt.xlabel("Run Index", fontsize=10)
        plt.ylabel("Final Path Cost", fontsize=10)
        plt.title("RRT* 2D - Final Cost for All Runs", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 3. Path Visualization (All and Representative)
    try:
        from twoD.environment import MapEnvironment
        from twoD.building_blocks import BuildingBlocks2D
        from twoD.visualizer import Visualizer

        MAP_DETAILS = {
            "json_file": "twoD/map_mp.json",
            "start": np.array([0.78, -0.78, 0.0, 0.0]),
            "goal": np.array([0.3, 0.15, 1.0, 1.1]),
        }
        planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
        bb = BuildingBlocks2D(planning_env)
        vis = Visualizer(bb)

        # --- Representative Visualization Section ---
        print("\n" + "=" * 50)
        print("REPRESENTATIVE VISUALIZATIONS (Closest to Mean Cost):")
        print("=" * 50)

        for p_bias in p_biases:
            # Filter valid runs
            valid_runs = [r for r in results if r.get("bias") == p_bias and r.get("path") is not None]
            if not valid_runs:
                continue

            # Identify the representative run
            mean_cost = np.mean([r["path_cost"] for r in valid_runs])
            # Find index of run with cost closest to mean
            idx = np.argmin([abs(r["path_cost"] - mean_cost) for r in valid_runs])
            rep_run = valid_runs[idx]

            print(f"\nBias {int(p_bias * 100)}% Representative:")
            print(f"  Run Index: {rep_run['run_index']} | Cost: {rep_run['path_cost']:.4f} (Mean: {mean_cost:.4f})")

            vis.visualize_plan(
                plan=np.array(rep_run["path"]),
                start=MAP_DETAILS["start"],
                goal=MAP_DETAILS["goal"]
            )

    except (ImportError, NameError):
        print("\nNote: Path visualization skipped. Ensure required classes are in your path.")



if __name__ == "__main__":
    # run_dot_2d_astar()
    # run_dot_2d_rrt()
    # run_dot_2d_rrt_star()
    # result_name=run_2d_rrt_motion_planning()
    # run_2d_rrt_inspection_planning()
    # run_2d_rrt_star_motion_planning()
    run_3d()
    # plot_results_run_3d()
    # find_best_path_3d()
    # rrt_2d_results("rrt_results_2d.json")
    # rrt_2d_inspection_results("rrt_inspection_results.json")
    # plot_2d_rrt_star_results()
