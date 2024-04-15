"""Example utility module.

Please use a file like this one to add extra functions.

"""
import yaml
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

LIFT_HEIGHT = 0.1
LAND_HEIGHT = 0.1


ASCENT_TIME = 1.2
DESCENT_TIME = 1.0
FIRST_SEGMENT_STEEPNESS = 0.5
THIRD_SEGMENT_STEEPNESS = 1.5

STEP = 0.3 #DISTANCE FROM GAT TO BUFFER WAYPOINTS
DUR_CURVE1 = 1.0
DUR_CURVE2 = 2
DUR_CURVE3 = 1.0




ADDITIONAL_OBS_BUFFER = 0.05
random.seed(1217)
SHOW_PLOTS = False

def extract_yaml():
    """
    Extract variables from YAML configuration file.
    Returns:
        dict: Dictionary containing the extracted variables.
    """

    # Read the YAML configuration file
    file_path = 'config.yaml'
    with open(file_path, 'r') as file:
        full_config = yaml.safe_load(file)
        cfg = full_config['quadrotor_config']
    return cfg

def get_trajectory():
    """
    Read a CSV file and return specific columns as arrays.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: Tuple containing arrays of t_scaled, ref_x, ref_y, and ref_z.
    """
    # Read the CSV file
    data = np.genfromtxt('trajectory.csv', delimiter=',', dtype=float)

    # Extract specific columns
    t_scaled = data[:, 0]
    ref_x = data[:, 1]
    ref_y = data[:, 2]
    ref_z = data[:, 3]
    return t_scaled, ref_x, ref_y, ref_z
    
def load_waypoints():
    cfg = extract_yaml()
    full_gates = np.array(cfg['gates'])[:, (0, 1, 5)]
    init_x = cfg['init_state']['init_x']
    init_y = cfg['init_state']['init_y']
    init_z = LIFT_HEIGHT
    endpoint = cfg['task_info']['stabilization_goal'] #len 3 list
    endpoint[2] = LAND_HEIGHT
    waypoints = [(init_x, init_y, init_z)]
    # heights = np.full((gates.shape[0], 1), 1.0)
    # gates = np.concatenate((gates, heights), axis=1)
    # for gate in gates:
    gate_height = 1.0
    for gate in full_gates:
        x, y, yaw = gate
        before_x = round(x + STEP * np.sin(yaw), 1)
        before_y = round(y - STEP * np.cos(yaw), 1)
        after_x = round(x - STEP * np.sin(yaw), 1)
        after_y = round(y + STEP * np.cos(yaw), 1)
        before_gate = [before_x, before_y, gate_height]
        at_gate = [x, y, gate_height]
        after_gate = [after_x, after_y, gate_height]
        waypoints.append(before_gate)
        waypoints.append(at_gate)
        waypoints.append(after_gate)

    waypoints.append(endpoint)
    waypoints = np.array(waypoints)
    # print(waypoints)
    return waypoints, LIFT_HEIGHT


def write_arrays_to_csv(array):
    with open('trajectory.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for row in array:
            writer.writerow(row)


def fit_curve(duration, offset, dt, points):
    # Extract x, y, and z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Fit 4th-degree polynomial to each dimension
    coefficients_x = np.polyfit(np.arange(len(points)), x, 4)
    coefficients_y = np.polyfit(np.arange(len(points)), y, 4)
    coefficients_z = np.polyfit(np.arange(len(points)), z, 4)

    # Generate points for the curve
    intervals = int(duration // dt)

    t = np.linspace(1, 2, intervals+1)
    x_interp = np.polyval(coefficients_x, t)
    y_interp = np.polyval(coefficients_y, t)
    z_interp = np.polyval(coefficients_z, t)
    time = np.arange(0, len(x_interp) * dt, dt)
    time = np.add(time, offset)
    traj = np.column_stack((time, x_interp, y_interp, z_interp))
    return np.round(traj, decimals=3)


def to_first_waypoint(dt, start_and_endpoints):
    start = start_and_endpoints[0]
    end = start_and_endpoints[1]

    # Step size for the additional points
    spacing = FIRST_SEGMENT_STEEPNESS

    # Add additional points
    extra_point1 = np.array([start[0], start[1], start[2] - spacing])
    extra_point2 = np.array([end[0] + spacing, end[1], end[2]])

    points = np.vstack((extra_point1, start, end, extra_point2))
    traj = fit_curve(ASCENT_TIME, 0, dt, points)
    print("\n\n\n\nascent from: ", start)
    print("ascent to from: ", end)
    
    return traj

def construct_obst_bounds(obst_cent):
    crazflie_width = 0.13
    noise = 0.2 + crazflie_width/2 + ADDITIONAL_OBS_BUFFER #obstacle can be moved by this much in x or y

    # Create a new 3D matrix
    obs_bounds = np.zeros((obst_cent.shape[0], obst_cent.shape[1], 2))

    # Iterate through each element of the original matrix and add/subtract the threshold
    for i in range(obst_cent.shape[0]):
        for j in range(obst_cent.shape[1]):
            obs_bounds[i, 0, j] = obst_cent[i, j] - noise
            obs_bounds[i, 1, j] = obst_cent[i, j] + noise
    return obs_bounds

def does_collide(coord_array, obs_bounds):
    is_colliding = np.any((obs_bounds[:, 0, 0] <= coord_array[0]) & 
                    (coord_array[0] <= obs_bounds[:, 1, 0]) &
                    (obs_bounds[:, 0, 1] <= coord_array[1]) & 
                    (coord_array[1] <= obs_bounds[:, 1, 1]))
    return is_colliding

def RRT_star_solver(start_time, dt, flat_waypoints, cfg):
    # flat_waypoints =np.array([[ 0.4, -2.5],
    #         [ 0.5, -2.5],
    #         [ 0.6, -2.5],
    #         [ 2. , -1.6],
    #         [ 2. , -1.5],
    #         [ 2. , -1.4],
    #         [ 0.1,  0.2],
    #         [ 0. ,  0.2],
    #         [-0.1,  0.2],
    #         [-0.5,  1.4],
    #         [-0.5,  1.5],
    #         [-0.5,  1.6]])
    obst_cent = np.array(cfg['obstacles'])[:, :2]
    obs_bounds = construct_obst_bounds(obst_cent)

    first_curve, second_start = shortest_curved_path(flat_waypoints[:6], DUR_CURVE1, start_time, dt, obs_bounds, include_end=False)

    second_curve, third_start = shortest_curved_path(flat_waypoints[3:9], DUR_CURVE2, (second_start+dt), dt, obs_bounds, include_end=False)

    third_curve, _ = shortest_curved_path(flat_waypoints[6:12], DUR_CURVE3, (third_start+dt), dt, obs_bounds, include_end=True)

    total_traj = np.vstack((first_curve, second_curve, third_curve))


    # value_to_test = np.array([1.4, -2.6])  # 2D value to test
    # collides = does_collide(value_to_test, obs_bounds)
    
    # print(collides)
    return total_traj

def shortest_curved_path(points, duration, time_offset, dt, obs_bounds, include_end):
    start_node_coord = points[2]
    end_node_coord = points[3]


    straight_line_traj = RRT_star_straight_lines(start_node_coord, end_node_coord, obs_bounds) #N x 2
    full_traj = np.vstack((points[:2], straight_line_traj, points[5:]))
    x_traj = full_traj[:, 0]
    y_traj = full_traj[:, 1]

    degree = 4
    coefficients_x = np.polyfit(np.arange(len(full_traj)), x_traj, degree)
    coefficients_y = np.polyfit(np.arange(len(full_traj)), y_traj, degree)

    # Generate the polynomial functions
    poly_function_x = np.poly1d(coefficients_x)
    poly_function_y = np.poly1d(coefficients_y)

    # Evaluate the polynomial functions to get the fitted x and y values
    intervals = int(duration // dt)
    if include_end:
        t = np.linspace(0, len(full_traj)-1, intervals+1)
        
    else:
        t = np.linspace(0, len(full_traj)-3, intervals+1)

    fitted_x = poly_function_x(t)
    fitted_y = poly_function_y(t)
    time = np.arange(0, len(fitted_x) * dt, dt)
    time = np.add(time, time_offset)
    last_time = time[-1]

    traj = np.column_stack((time, fitted_x, fitted_y))
    return np.round(traj, decimals=3), last_time


def distance_thresh(reference_point, test_point, thresh): #Returns True if 2 nodes are within thresh distance
    x1, y1 = reference_point
    x2, y2 = test_point
    dx = x2 - x1
    dy = y2 - y1
    if (abs(dx) > thresh or abs(dy) > thresh): #Dont calculate irrelevent distances
        return False
    dist = math.sqrt((dx)**2 + (dy)**2)
    return dist <= thresh

def get_child_nodes(node_list, given_node): # Returns list of all child nodes
    child_nodes = []
    for index, node in enumerate(node_list):
        if node.parent:
            if given_node.is_state_identical(node.parent):
                child_nodes.append((index, node))
    return child_nodes

def fast_propogate(previous_parent_cost, new_parent_node, child_node): # Change parent node and update cost
    previous_child_cost = child_node.cost
    last_cost = previous_child_cost - previous_parent_cost
    child_node.cost = new_parent_node.cost + last_cost
    child_node.parent = new_parent_node
    return child_node, previous_child_cost

def propagate_cost_change(node_list, previous_parent_cost, node): #Update costs for all downstream nodes after rewiring
    for index, child_node in get_child_nodes(node_list, node):
        updated_child_node, previous_child_cost = fast_propogate(previous_parent_cost, node, child_node)
        node_list[index] = updated_child_node
        node_list = propagate_cost_change(node_list, previous_child_cost, updated_child_node)
    return node_list


def gaussian_sampling_between_points(point1, point2, std_dev):
    # Calculate the equation of the line passing through the points
    # y = mx + c
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])
    c = point1[1] - m * point1[0]

    # Sample a single point along the line using Gaussian distribution
    x_value = np.random.uniform(point1[0], point2[0])
    y_value = m * x_value + c + np.random.normal(0, std_dev)

    # Combine x and y values into a point
    sampled_point = np.array([x_value, y_value])
    return sampled_point


class Node():
    def __init__(self, coords, cost=0, short_cost=0, parent=None, path_x=None, path_y=None):
        self.x, self.y = coords
        self.cost = cost
        self.short_cost = short_cost
        self.parent = parent
        self.path_x = path_x
        self.path_y = path_y


    def propogate(self, node):
        elements_in_path = 20
        self.path_x = np.linspace(node.x, self.x, elements_in_path, endpoint=True)
        self.path_y = np.linspace(node.y, self.y, elements_in_path, endpoint=True)
        self.parent = node
        dx = self.x - node.x
        dy = self.y - node.y
        self.short_cost = math.sqrt((dx)**2 + (dy)**2)
        self.cost = node.cost + self.short_cost

    def valid_path(self, obs):
        for ptx, pty in zip(self.path_x, self.path_y):
            value_to_test = np.array([ptx, pty])  # 2D value to test
            collides = does_collide(value_to_test, obs)
            if collides:
                return False
        return True
    def is_state_identical(self, node):
        dx = abs(self.x - node.x)
        dy = abs(self.y - node.y)
        thresh = 0.0001
        if dx < thresh and dy <thresh:
            return True
        else:
            return False
    def copy(self):
        return Node((self.x, self.y), self.cost, self.short_cost, self.parent, self.path_x, self.path_y)


def RRT_star_straight_lines(start_node_coord, end_node_coord, obs_bounds):

    ##TUNABLE PARAMETERS
    STEP_SIZE = 0.1 #Step size to take in direction of random point for new node
    BIAS = 0.1 #Probability of random point at the goal
    NEIGHBOUR_RADIUS = 1.0 #Size of radius to check for neighbour rewiring
    FIRST_RETURN_ITER = 300 #Try to return after X many iterations if a solution was found
    ITER_STEP_SIZE = 100 #If no route found at FIRST_RETURN_ITER, check again every ITER_STEP_SIZE iterations
    MAX_ITER = 500 #Maximum number of nodes to try making    
            
    ##INITIALIZE VARIABLES
    if MAX_ITER > FIRST_RETURN_ITER:
        break_iters = list(range(FIRST_RETURN_ITER, MAX_ITER+1, ITER_STEP_SIZE))
    else:
        break_iters = [MAX_ITER]
    return_idx = 0
    is_first_time_at_goal = True
    final_node_index = -1
    i = 0

    node_list = [Node(start_node_coord)]

    #Line between start and end
    m = (end_node_coord[1] - start_node_coord[1]) / (end_node_coord[0] - start_node_coord[0])
    c = start_node_coord[1] - m * start_node_coord[0]
    std_dev = 0.3

    goal_node = Node((end_node_coord[0], end_node_coord[1]))

    #MAIN WHILE LOOP
    while i < MAX_ITER: #loop through each new iteration
        if final_node_index != -1 and i >= break_iters[return_idx]:
            break
        i += 1
        if ((i-1)%ITER_STEP_SIZE == 0 and i > break_iters[return_idx]): #update next iteration to check if path found
            return_idx += 1 

        # Generate a random vehicle state (x, y, yaw) 1 random sample somewhere
        prob = random.uniform(0, 1)
        if prob < BIAS: #Bias towards goal
            x2, y2 = end_node_coord[0], end_node_coord[1]
            # at_end = True
        else:
            # Sample a single point along the line using Gaussian distribution
            x2 = np.random.uniform(start_node_coord[0], end_node_coord[0])
            y2 = m * x2 + c + np.random.normal(0, std_dev)
        min_dis = float('inf')
        
        for n in node_list: #find closest node
            # if (n.is_state_identical(goal_node)):
            #     continue
            dx = x2 - n.x
            dy = y2 - n.y

            if (abs(dx) > min_dis or abs(dy) > min_dis): #Dont calculate irrelevent distances
                continue #next iter of for loop

            distance = math.sqrt((dx)**2 + (dy)**2)

            if STEP_SIZE < distance < min_dis: #new node at STEP_SIZE distance
                closest_node = n
                min_dis = distance
                step_x = dx / distance * STEP_SIZE
                step_y = dy / distance * STEP_SIZE
            elif STEP_SIZE > distance and min_dis > distance: #if close to existing node, just make new node at the random point without STEP
                closest_node = n
                min_dis = distance
                step_x = dx
                step_y = dy
        
        #sample point is defined
        samp_x = closest_node.x + (step_x)
        samp_y = closest_node.y + (step_y)
        samp_node = Node((samp_x, samp_y))
        if samp_node.is_state_identical(closest_node):
            continue
        #initialize parameters to find nearby nodes
        neighbour_list = []
        min_cost = float('inf')
        thresh_cost = float('inf')
        not_valid_node = True
        index_of_neighbours = 0

        for node_index, n in enumerate(node_list): #get all neighbours and find closest
            if distance_thresh((samp_x, samp_y), (n.x, n.y), NEIGHBOUR_RADIUS):
                neighbour_list.append((node_index, n))
                
                if distance_thresh((samp_x, samp_y), (n.x, n.y), thresh_cost):
                    # trial_node = rrt_dubins.propogate(n, samp_node) 
                    trial_node = Node((samp_x, samp_y))
                    trial_node.propogate(n)
                    cost = trial_node.cost
                    if cost < min_cost:
                        if trial_node.valid_path(obs_bounds): #find parent with lowest cost and valid trajectory
                            not_valid_node = False
                            min_cost = cost
                            thresh_cost = trial_node.short_cost
                            best_node_config = trial_node.copy()
                            best_parent_index = index_of_neighbours
                            # index_of_neighbours += 1
                index_of_neighbours += 1

        if not_valid_node: #bad node where trajectories not valid
            continue
        node_list.append(best_node_config) #add new node with parent
        del neighbour_list[best_parent_index]

        for node_index, child_n in neighbour_list: #rewiring of other neighbours with new node as parent if applicable
            if child_n.is_state_identical(best_node_config.parent): 
                continue
            curr_cost = child_n.cost
            max_viable_cost = curr_cost - best_node_config.cost #difference from other node's current cost vs new node's cost
            if distance_thresh((best_node_config.x, best_node_config.y), (child_n.x, child_n.y), max_viable_cost): #if potentially close enough
                # new_child = rrt_dubins.propogate(best_node_config, child_n)
                new_child = child_n.copy()
                new_child.propogate(best_node_config)
                new_cost = new_child.cost
                if new_cost < curr_cost: #if lower cost with rewiring
                    if new_child.valid_path(obs_bounds):
                        node_list[node_index] = new_child
                        node_list = propagate_cost_change(node_list, curr_cost, new_child) #rewire and propogate costs
        
        goal_dx = samp_x - end_node_coord[0]
        goal_dy = samp_y - end_node_coord[1]

        proximity = math.sqrt((goal_dx)**2 + (goal_dy)**2) #calculate new nodes distance to goal
        # Check if new_node is close to goal for the first time
        if (proximity < STEP_SIZE and is_first_time_at_goal):
            if best_node_config.is_state_identical(goal_node):
                final_node_index = len(node_list) - 1 #index of the node at the goal position
                is_first_time_at_goal = False
            else:
                
                final_node = Node((end_node_coord[0], end_node_coord[1]))
                final_node.propogate(best_node_config)
                if not final_node.valid_path(obs_bounds): #crash
                    continue
                node_list.append(final_node)
                final_node_index = len(node_list) - 1 #index of the node at the goal position
                is_first_time_at_goal = False

    if final_node_index != -1: #If a valid path was found
        temp_node = node_list[final_node_index]
        node_list = [temp_node]
        while temp_node.parent: #create list of all parents, starting from goal node
            temp_node = temp_node.parent
            node_list.insert(0, temp_node)
        coords_x = []
        coords_y = []
        for n in node_list:
            if not n.path_x is None: #start node has no path preceding it
                coords_x.extend(n.path_x.tolist())
                coords_y.extend(n.path_y.tolist())
        
        coords = np.column_stack((coords_x, coords_y))
        coords = np.unique(coords, axis=0)
        if start_node_coord[0] > end_node_coord[0]: coords = coords[::-1]
        print("start: ", coords[0])
        print("end: ", coords[-1])
        print("\n\n")

        if SHOW_PLOTS:
            # Plot obstacles
            for obstacle in obs_bounds:
                bottom_left = obstacle[0]
                top_right = obstacle[1]
                width = top_right[0] - bottom_left[0]
                height = top_right[1] - bottom_left[1]
                rectangle = plt.Rectangle((bottom_left[0], bottom_left[1]), width, height, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rectangle)

            # Plot scatter points
            plt.scatter([point[0] for point in coords], [point[1] for point in coords], color='blue')
            # Set plot limits
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            # Set aspect ratio to be equal
            plt.gca().set_aspect('equal', adjustable='box')
            # Add labels and title
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Rectangular Obstacles and Scatter Points')

            # Show plot
            plt.grid(True)
            plt.show()


        return coords
    print("No route found with given max_iter, consider increasing")
    return None
    



def to_ground(start_time, dt, start_and_endpoints):
    start = start_and_endpoints[0]
    end = start_and_endpoints[1]
    print("descent from: ", start)
    print("descent to from: ", end, "\n\n\n\n")

    # Step size for the additional points
    spacing = THIRD_SEGMENT_STEEPNESS

    # Add additional points
    extra_point1 = np.array([start[0], start[1]-spacing, start[2]])
    extra_point2 = np.array([end[0] + spacing, end[1], end[2]-spacing])

    points = np.vstack((extra_point1, start, end, extra_point2))
    traj = fit_curve(DESCENT_TIME, start_time, dt, points)
    return traj

def determine_trajectory():
    waypoints, _ = load_waypoints()
    flat_waypoints = waypoints[1:-1, :2]
    cfg = extract_yaml()
    ctrl_freq = cfg['ctrl_freq']
    dt = 1.0/float(ctrl_freq)
    start_3d_traj = to_first_waypoint(dt, waypoints[0:2])
    last_time_p1 = start_3d_traj[-1, 0] #latest_time
    start_time = last_time_p1 + dt
    print("\n\n\n")

    middle_2d_traj = RRT_star_solver(start_time, dt, flat_waypoints, cfg)
    last_time_p2 = middle_2d_traj[-1, 0]
    start_time = last_time_p2 + dt
    end_3d_traj = to_ground(start_time, dt, waypoints[-2:])

    
    height_column = np.ones((middle_2d_traj.shape[0], 1), dtype=float)
    middle_2d_traj = np.hstack((middle_2d_traj, height_column))

    full_traj = np.vstack((start_3d_traj, middle_2d_traj, end_3d_traj))
    write_arrays_to_csv(full_traj)


if __name__ == '__main__':
    determine_trajectory()
    # load_waypoints()