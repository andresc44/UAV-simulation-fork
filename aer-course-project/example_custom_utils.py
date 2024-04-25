"""Example utility module.

Please use a file like this one to add extra functions.

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import itertools
from Planners.rrt_star_planner import rrt_star_planner
from Planners.dubins_path_problem import RRT_dubins_problem, check_path, get_path

show_animation = True
gates = np.array([  # x, y, z, r, p, y, type
    [0.5, -2.5, 0, 0, 0, -1.57, 0],  # gate 1
    [2.0, -1.5, 0, 0, 0, 0, 0],  # gate 2
    [0.0, 0.2, 0, 0, 0, 1.57, 0],  # gate 3
    [-0.5, 1.5, 0, 0, 0, 0, 0]])  # gate 4])
obstacles = np.array([  # x, y, z, r, p, y
    [1.5, -2.5, 0, 0, 0, 0],  # obstacle 1
    [0.5, -1.0, 0, 0, 0, 0],  # obstacle 2
    [1.5, 0, 0, 0, 0, 0],  # obstacle 3
    [-1.0, 0, 0, 0, 0, 0]])  # obstacle 4
gate_rad = 0.1
gate_width = 0.2
obstacles_rad = 0.15
obstacles_noise = 0.2
obs_rad_corrupted = obstacles_rad + obstacles_noise
map_bound = [-2.00, 3.00, -3.50, 2.50]  # min_x max_x min_y max_y
start = np.array([-1.00, -2.00, 0.00])
final_goal = np.array([2.0, 1.00, 0.00])
plot_rrt_planning = False


def main():
    orders = np.array(list(itertools.permutations([1, 2, 3, 4])))
    orders = [[4, 2, 3, 2]]
    for order in orders:
        compute_fullpath(order)
    pass


def compute_fullpath(gate_order):
    file_name = f"Path_files_rrt_star/path_{gate_order[0]}{gate_order[1]}{gate_order[2]}{gate_order[3]}/"

    if not os.path.isdir(file_name):
        os.makedirs(file_name)

    obs = []
    previous_gate = -1
    # Add gates boarders as obstacles
    repeated_gate = False
    for gate in gates:
        # if previous_gate == gate:
        #     temp_gate = gate
        #     repeated_gate = True
        #     if temp_gate[5]==0:
        #         temp_gate[1] = temp_gate[1] + 2 * gate_width
        #     else:
        #         temp_gate[0] = temp_gate[0] + 2 * gate_width
        #     temp_gate[5] = temp_gate[5] + math.pi
        #     # for vertical gate
        #     if temp_gate[5] != 0:
        #         obs.append([temp_gate[0], temp_gate[1] + gate_width, gate_rad])
        #         obs.append([temp_gate[0], temp_gate[1] - gate_width, gate_rad])
        #     # for horizontal gate
        #     else:
        #         obs.append([temp_gate[0] + gate_width, temp_gate[1], gate_rad])
        #         obs.append([temp_gate[0] - gate_width, temp_gate[1], gate_rad])


        # for vertical gate
        if gate[5] != 0:
            obs.append([gate[0], gate[1] + gate_width, gate_rad])
            obs.append([gate[0], gate[1] - gate_width, gate_rad])
        # for horizontal gate
        else:
            obs.append([gate[0] + gate_width, gate[1], gate_rad])
            obs.append([gate[0] - gate_width, gate[1], gate_rad])
        previous_gate = gate

    # Add actual obstacles
    for obstacle in obstacles:
        obs.append([obstacle[0], obstacle[1], obs_rad_corrupted])

    obs = np.array(obs)
    temp_start = start
    temp_goals = gates[[gate - 1 for gate in gate_order]][:, [0, 1, 5]]
    temp_goals[:, 2] = temp_goals[:, 2] + np.pi/2
    temp_goals = np.vstack((temp_goals, final_goal))
    full_path = np.empty((0, 3), float)

    for i, temp_goal in enumerate(temp_goals):

        print("temp_start:", temp_start)
        print("temp_goal:", temp_goal)

        rrt_dubins = RRT_dubins_problem(start=temp_start, goal=temp_goal,
                                        obstacle_list=obs,
                                        map_area=map_bound,
                                        max_iter=1000,
                                        curvature=1 / 0.4)
        path_node_list = rrt_star_planner(rrt_dubins, plot_rrt_planning)
        is_path_valid = check_path(rrt_dubins, path_node_list)
        path = np.array(get_path(path_node_list))
        if not is_path_valid or len(path) == 0:
            print(f'Test Failed: Given path is not valid\n Visualize the path to debug')

        rx = path[:,0]
        ry = path[:,1]
        rt = path[:,2]
        full_path = np.vstack((full_path, path))
        np.save(file_name+f'test_path{i}.npy', path)    # .npy extension is added if not given

        if plot_rrt_planning:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            plt.savefig(file_name + f'test_path{i}.png', bbox_inches='tight', dpi=400)
            plt.show(block=False)

        temp_start = path[-1, 0:3]

    np.save(file_name + 'test_path_final.npy', full_path)

    if show_animation:  # pragma: no cover
        plt.plot(full_path[:,0], full_path[:,1], "-r")

        for (ox, oy, size) in obs:
            plot_circle(ox, oy, size)

        plt.pause(0.001)
        plt.savefig(file_name + f'test_path_full.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)


def plot_circle(x, y, size, color="-b"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)
if __name__ == '__main__':
    main()
