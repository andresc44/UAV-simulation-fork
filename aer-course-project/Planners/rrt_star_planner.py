"""
Assignment #2 Template file
"""
import copy
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from Planners import dubins_path_planning

# try:
#     import dubins_path_planning
# except ImportError:
#     raise ImportError("Missing planner for dubins_path_problem")

continue_after_goal_reached = False
maximum_iteration_with_goal_reached = 250
step_size = 0.01
GlobalSearch = True
verbose = True
exploration_radius = 10
"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees* (RRT)
for the problem setup given by the RRT_DUBINS_PROBLEM class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file rrt_star_planner.py. Your
   implementation can be tested by running RRT_DUBINS_PROBLEM.PY (check the 
   main function).
2. Read all class and function documentation in RRT_DUBINS_PROBLEM carefully.
   There are plenty of helper function in the class to ease implementation.
3. Your solution must meet all the conditions specified below.
4. Below are some do's and don't for this problem as well.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. The solution loop must not run for more that a certain number of random iterations
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out-related issues and will be set generously.
2. The planning function must return a list of nodes that represent a collision-free path
   from start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must define a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation for the node class to understand the terminology)
3. The returned path should have the start node at index 0 and goal node at index -1,
   while the parent node for node i from the list should be node i-1 from the list, ie,
   the path should be a valid list of nodes.
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   RRT_DUBINS_PROBLEM.map_area.

DO(s) and DONT(s)
-------------------
1. Do not rename the file rrt_star_planner.py for submission.
2. Do not change change the PLANNING function signature.
3. Do not import anything other than what is already imported in this file.
4. You can write more function in this file in order to reduce code repetition
   but these function can only be used inside the PLANNING function.
   (since only the planning function will be imported)
"""


def rrt_star_planner(rrt_dubins, display_map=False):
    """
        Execute RRT* planning using Dubins-style paths. Make sure to populate the node_list.

        Inputs
        -------------
        rrt_dubins  - (RRT_DUBINS_PROBLEM) Class containing the planning
                      problem specification
        display_map - (boolean) flag for animation on or off (OPTIONAL)

        Outputs
        --------------
        (list of nodes) This must be a valid list of connected nodes that form
                        a path from start to goal node

        NOTE: In order for rrt_dubins.draw_graph function to work properly, it is important
        to populate rrt_dubins.nodes_list with all valid RRT nodes.
    """
    i = 0
    goal_reached = False
    rrt_dubins.start.parent = -1
    rrt_dubins.goal.parent = -1
    best_path_cost = float("inf")
    best_path_node_indices = []
    while (i < rrt_dubins.max_iter) and (
            (not goal_reached) or (continue_after_goal_reached and (i < maximum_iteration_with_goal_reached))):

        if verbose:
            print(f"Iteration {i} of {rrt_dubins.max_iter} - number of nodes = {len(rrt_dubins.node_list)}")
        i += 1
        # Generate a random vehicle state (x, y, yaw)
        random.randint(0, 100)
        rand_node = rrt_dubins.Node(x=random.uniform(rrt_dubins.x_lim[0], rrt_dubins.x_lim[1]),
                                    y=random.uniform(rrt_dubins.y_lim[0], rrt_dubins.y_lim[1]),
                                    yaw=random.uniform(-np.pi, np.pi))
        # Find an existing node nearest to the random vehicle state
        dist_list = [np.sqrt((rand_node.x - node.x) ** 2 + (rand_node.y - node.y) ** 2) for node in
                     rrt_dubins.node_list]
        close_nodes_indices = np.where(np.array(dist_list) < (exploration_radius + 2 * rrt_dubins.curvature))[0]
        if len(close_nodes_indices) < 1:
            closest_node_index = np.argmin(dist_list)
            new_node = create_new_node(rrt_dubins=rrt_dubins, to_node=rand_node, curvature=rrt_dubins.curvature,
                                       parent_node_index=closest_node_index)
        else:
            least_cost = float("inf")
            for node_index in close_nodes_indices:
                if dist_list[node_index] > least_cost:
                    continue
                potential_node = create_new_node(rrt_dubins=rrt_dubins, to_node=rand_node,
                                                 curvature=rrt_dubins.curvature,
                                                 parent_node_index=node_index)
                if potential_node.cost < least_cost:
                    new_node = potential_node
                    least_cost = new_node.cost

        # Draw current view of the map
        # PRESS ESCAPE TO EXIT
        # if display_map and (((i+1) % 50) == 0):
        #     rrt_dubins.draw_graph()

        # Check if the path between nearest node and random state has obstacle collision
        # Add the node to nodes_list if it is valid
        if check_collision(rrt_dubins, new_node):
            rrt_dubins.node_list.append(new_node)  # Storing all valid nodes
            # Check if a collision free path exist between new node and goal
            goal_node = create_new_node(rrt_dubins=rrt_dubins, to_node=rrt_dubins.goal, curvature=rrt_dubins.curvature,
                                        parent_node_index=len(rrt_dubins.node_list) - 1)
            if check_collision(rrt_dubins, goal_node):
                if goal_node.cost < best_path_cost:
                    rrt_dubins.node_list.append(goal_node)  # Storing all valid nodes
                    rrt_dubins.goal = goal_node
                    goal_reached = True
                    best_path_node_indices = generate_final_course(rrt_dubins=rrt_dubins)
                    best_path_cost = goal_node.cost

                    if verbose:
                        print(f"This is a new best path for the goal")

                if verbose:
                    print(f"Goal reached! number of iterations: {i} - number of nodes: {len(rrt_dubins.node_list)}")

                # Draw current view of the map
                # PRESS ESCAPE TO EXIT
                if display_map:
                    rrt_dubins.draw_graph()

    if i == rrt_dubins.max_iter:

        if verbose:
            print(f'reached max iterations - goal reached? {goal_reached}')
    if goal_reached:
        return [rrt_dubins.node_list[node_index] for node_index in reversed(best_path_node_indices)]
    else:
        return


def generate_final_course(rrt_dubins):
    path_node_indices = [len(rrt_dubins.node_list) - 1]
    node = copy.deepcopy(rrt_dubins.goal)
    while node.parent >= 0:
        path_node_indices.append(node.parent)
        # for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
        node = rrt_dubins.node_list[node.parent]
    return path_node_indices


def create_new_node(rrt_dubins, to_node, curvature, parent_node_index):
    from_node = copy.deepcopy(rrt_dubins.node_list[parent_node_index])
    new_node = copy.deepcopy(from_node)
    # best_cost, path_x, path_y, path_yaw = get_cost_and_path(from_node, to_node, curvature)
    path_x, path_y, path_yaw, mode, best_cost = dubins_path_planning.dubins_path_planning(from_node.x, from_node.y,
                                                                                          from_node.yaw, to_node.x,
                                                                                          to_node.y, to_node.yaw,
                                                                                          curvature)
    new_node.x = to_node.x
    new_node.y = to_node.y
    new_node.yaw = to_node.yaw
    new_node.cost += best_cost
    new_node.parent = parent_node_index
    new_node.path_x = path_x
    new_node.path_y = path_y
    new_node.path_yaw = path_yaw

    return new_node


def get_cost_and_path(from_node, to_node, curvature):
    dist_x = to_node.x - from_node.x
    dist_y = to_node.y - from_node.y

    delta_x = math.cos(from_node.yaw) * dist_x + math.sin(from_node.yaw) * dist_y
    delta_y = - math.sin(from_node.yaw) * dist_x + math.cos(from_node.yaw) * dist_y
    delta_yaw = to_node.yaw - from_node.yaw
    capital_d = math.hypot(delta_x, delta_y)
    d = capital_d * curvature

    theta = dubins_path_planning.mod2pi(math.atan2(delta_y, delta_x))
    alpha = dubins_path_planning.mod2pi(- theta)
    beta = dubins_path_planning.mod2pi(delta_yaw - theta)

    planners = [dubins_path_planning.left_straight_left, dubins_path_planning.right_straight_right,
                dubins_path_planning.left_straight_right,
                dubins_path_planning.right_straight_left, dubins_path_planning.right_left_right,
                dubins_path_planning.left_right_left]

    best_cost = float("inf")
    bt, bp, bq, best_mode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))
        if best_cost > cost:
            bt, bp, bq, best_mode = t, p, q, mode
            best_cost = cost
    lengths = [bt, bp, bq]

    px, py, pyaw, directions = dubins_path_planning.generate_local_course(
        sum(lengths), lengths, best_mode, curvature, step_size)

    return best_cost, px, py, pyaw


def check_collision(rrt_dubins, new_node):
    if new_node is None:
        return False

    node = copy.deepcopy(new_node)
    path_nodes = [new_node]
    while node.parent > 0:
        node = rrt_dubins.node_list[node.parent]
        path_nodes.append(node)

    x_list = np.hstack([[node_x for node_x in node.path_x] for node in path_nodes])
    y_list = np.hstack([[node_y for node_y in node.path_y] for node in path_nodes])

    for (ox, oy, size) in rrt_dubins.obstacle_list:
        dx_list = [ox - x for x in x_list]
        dy_list = [oy - y for y in y_list]
        d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

        if min(d_list) <= size ** 2:
            return False  # collision

    # Check if we are within the map
    if any(x_list < rrt_dubins.x_lim[0]) or any(x_list > rrt_dubins.x_lim[1]) or any(
            y_list < rrt_dubins.y_lim[0]) or any(
            y_list > rrt_dubins.y_lim[1]):
        return False  # Outside boundaries

    return True  # safe
