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
from scipy.integrate import quad

ORDER = [1, 3, 4, 2, 1, 4] # Gate Order, Slightly slower for 4, 3, 2, 1
DRONE_SPEED = 0.97 #m/s
STEP = 0.4 #DISTANCE FROM GATE TO BUFFER WAYPOINTS

GATE1YDIFF = 0.075
GATE2XDIFF = -0.13
GATE3YDIFF = -0.08
GATE4XDIFF = -0.01
# GATE1YDIFF = 0.0
# GATE2XDIFF = 0.0
# GATE3YDIFF = 0.0
# GATE4XDIFF = 0.0

ASCENT_RADIUS = 0.25 # How far away to be when reaching target of 1m
POLY_DEGREE = 7 #How much to fit to RRT* path
GATE_STRAIGHT_WEIGHT = 30 #How vital is it to go more straight during gate
#Above should be even number

ADDITIONAL_OBS_BUFFER = 0.03 #Pretty high, could lower
GATE_BUFFER = 0.1 #How to treat as obstacle, could increase

LIFT_HEIGHT = 0.1 #How high to go up vertically
FIRST_SEGMENT_STEEPNESS = 0.5 # Don't change

##TUNABLE PARAMETERS RRT*
STEP_SIZE = 0.3 #Step size to take in direction of random point for new node
BIAS = 0.2 #Probability of random point at the goal
NEIGHBOUR_RADIUS = 1.0 #Size of radius to check for neighbour rewiring
FIRST_RETURN_ITER = 500 #Try to return after X many iterations if a solution was found
ITER_STEP_SIZE = 1000 #If no route found at FIRST_RETURN_ITER, check again every ITER_STEP_SIZE iterations
MAX_ITER = 4000 #Maximum number of nodes to try making  
STD_DEV_SAMPLING = 0.7

random.seed(1217)
SHOW_PLOTS = False
SHOW_SAMPLES_PLOTS = False

def extract_yaml(yaml_path):
    """
    Extract variables from YAML configuration file.
    Returns:
        dict: Dictionary containing the extracted variables.
    """

    # Read the YAML configuration file
    with open(yaml_path, 'r') as file:
        full_config = yaml.safe_load(file)
        cfg = full_config['quadrotor_config']
    return cfg

def get_trajectory(csv_file):
    """
    Read a CSV file and return specific columns as arrays.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: Tuple containing arrays of t_scaled, ref_x, ref_y, and ref_z.
    """
    # Read the CSV file
    data = np.genfromtxt(csv_file, delimiter=',', dtype=float)

    # Extract specific columns
    t_scaled = data[:, 0]
    ref_x = data[:, 1]
    ref_y = data[:, 2]
    ref_z = data[:, 3]
    return t_scaled, ref_x, ref_y, ref_z
    
def write_arrays_to_csv(csv_path, array):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for row in array:
            writer.writerow(row)

def fit_curve(offset, dt, points):
    # Extract x, y, and z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    degree = POLY_DEGREE
    # Fit 4th-degree polynomial to each dimension
    coefficients_x = np.polyfit(np.arange(len(points)), x, degree)
    coefficients_y = np.polyfit(np.arange(len(points)), y, degree)
    coefficients_z = np.polyfit(np.arange(len(points)), z, degree)

    t1 = 1.0
    t2 = 2.0
    
      # Example value for t2
    poly_function_x = np.poly1d(coefficients_x)
    poly_function_y = np.poly1d(coefficients_y)
    poly_function_z = np.poly1d(coefficients_z)
    poly_derivative_x = np.polyder(poly_function_x)
    poly_derivative_y = np.polyder(poly_function_y)
    poly_derivative_z = np.polyder(poly_function_z)
    

    # Define a function to calculate the magnitude of velocity at a given time
    def velocity_magnitude(t):
        vx = poly_derivative_x(t)
        vy = poly_derivative_y(t)
        vz = poly_derivative_z(t)
        return np.sqrt(vx**2 + vy**2 + vz**2)


    # Integrate the velocity magnitude function over the time interval
    total_distance, _ = quad(velocity_magnitude, t1, t2)

    duration = total_distance / DRONE_SPEED
    print("ascent distance: ", total_distance, "duration: ", duration)

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
    end = start_and_endpoints[2]

    # Step size for the additional points
    spacing = FIRST_SEGMENT_STEEPNESS

    # Add additional points
    extra_point1 = np.array([start[0], start[1], start[2] - spacing])
    extra_point2 = np.array([end[0] + spacing, end[1], end[2]])

    points = np.vstack((extra_point1, start, end, extra_point2))
    traj = fit_curve(0, dt, points)
    print("\n\n\n\nascent from: ", start)
    print("ascent to from: ", end)
    
    return traj

# def to_ground(start_time, dt, start_and_endpoints):
#     start = start_and_endpoints[0]
#     end = start_and_endpoints[1]
#     print("descent from: ", start)
#     print("descent to from: ", end, "\n\n\n\n")

#     # Step size for the additional points
#     spacing = THIRD_SEGMENT_STEEPNESS

#     # Add additional points
#     extra_point1 = np.array([start[0], start[1]-spacing, start[2]])
#     extra_point2 = np.array([end[0] + spacing, end[1], end[2]-spacing])

#     points = np.vstack((extra_point1, start, end, extra_point2))
#     traj = fit_curve(DESCENT_TIME, start_time, dt, points)
#     return traj

def get_distance(prior, dest):
    x_p, y_p = prior
    x_d, y_d = dest
    dx = x_d - x_p
    dy = y_d - y_p
    dist = math.sqrt((dx)**2 + (dy)**2)
    return dist

class Gate():
    def __init__(self, row_info, gate_id):
        self.row_info = row_info
        self.gate_id = gate_id
        self.reversed = False
        x, y, yaw = row_info
        self.centre = (x, y)
        self.yaw = yaw
        before_x = round(x + STEP * np.sin(yaw), 1)
        before_y = round(y - STEP * np.cos(yaw), 1)
        after_x = round(x - STEP * np.sin(yaw), 1)
        after_y = round(y + STEP * np.cos(yaw), 1)
        self.before = (before_x, before_y)
        self.after = (after_x, after_y)

    def apply_reverse(self):
        temp = self.after
        self.after = self.before
        self.before = temp
        self.reversed = True
        return
    def copy(self):
        return Gate(self.row_info, self.gate_id)

def order_waypoints(startpoint, endpoint, ref_gates, obs_bounds):
    min_dist = float('inf')
    # init_z = LIFT_HEIGHT
    # endpoint[2] = LAND_HEIGHT
    combinations = 2 ** (len(ref_gates))
    for i in range(combinations):
        binary_mask = format(i, f'0{len(ref_gates)}b')  # Convert the integer to binary_mask with 4 bits
        reverse_bool_array = np.array(list(map(int, list(binary_mask))))
        temp_gates = []
        for j in range(len(ref_gates)):
            bool_gate = ref_gates[j].copy()
            if reverse_bool_array[j] == 1:
                bool_gate.apply_reverse()
            temp_gates.append(bool_gate)
        total_dist = get_distance(startpoint, temp_gates[0].before)

        for j in range(len(ref_gates)-1):
            total_dist += get_distance(temp_gates[j].after, temp_gates[j+1].before)
        
        total_dist += get_distance(temp_gates[-1].after, endpoint)
        if total_dist < min_dist:
            min_dist = total_dist
            best_gate_orientation = temp_gates
    
    first_target = best_gate_orientation[0].before
    first_RRT_traj = RRT_star_straight_lines(startpoint, first_target, obs_bounds)

    first_point_cushion = 0.2
    for coordinates in first_RRT_traj:
        cur_dist =  get_distance(startpoint, (coordinates[0], coordinates[1]))
        if cur_dist< ASCENT_RADIUS: ascent_coordinates = coordinates
        if cur_dist< ASCENT_RADIUS+first_point_cushion: first_after = coordinates
        if cur_dist< ASCENT_RADIUS+first_point_cushion*2: sec_after = coordinates
    ascent_coordinates = np.round(ascent_coordinates, decimals=3)
    first_after = np.round(first_after, decimals=3)
    sec_after = np.round(sec_after, decimals=3)
    
    waypoints = [list(startpoint)]
    
    waypoints.extend([ascent_coordinates, first_after, sec_after])

    for gate in best_gate_orientation:
        waypoints.extend([gate.before, gate.centre, gate.after])

    waypoints.append(endpoint)
    waypoints = np.array(waypoints)
    ones_col = np.ones(waypoints.shape[0]).reshape((waypoints.shape[0], 1))
    waypoints = np.hstack((waypoints, ones_col))
    waypoints[0, 2] = LIFT_HEIGHT
    # print("waypoints: \n", waypoints)
    return waypoints

def load_waypoints(yaml_path):
    order=ORDER
    order = np.array(order)-1
    cfg = extract_yaml(yaml_path)
    full_gates = np.array(cfg['gates'])[:, (0, 1, 5)]
    print("full gates before: \n", full_gates)
    full_gates[0][1] = full_gates[0][1] + GATE1YDIFF
    full_gates[1][0] = full_gates[1][0] + GATE2XDIFF
    full_gates[2][1] = full_gates[2][1] + GATE3YDIFF
    full_gates[3][0] = full_gates[3][0] + GATE4XDIFF
    print("full gates after: \n", full_gates)
    init_x = cfg['init_state']['init_x']
    init_y = cfg['init_state']['init_y']
    # init_z = LIFT_HEIGHT
    endpoint = cfg['task_info']['stabilization_goal'][:2] #originally len 3 list
    # endpoint[2] = LAND_HEIGHT
    startpoint = (init_x, init_y)
    # waypoints = [(init_x, init_y, init_z)]
    # heights = np.full((gates.shape[0], 1), 1.0)
    # gates = np.concatenate((gates, heights), axis=1)
    # for gate in gates:
    gate_height = 1.0
    gate_obs = []
    gate_half_width = 0.22


    for gate in full_gates:
        x, y, yaw = gate
        # before_x = round(x + STEP * np.sin(yaw), 1)
        # before_y = round(y - STEP * np.cos(yaw), 1)
        # after_x = round(x - STEP * np.sin(yaw), 1)
        # after_y = round(y + STEP * np.cos(yaw), 1)
        # before_gate = [before_x, before_y, gate_height]
        # at_gate = [x, y, gate_height]
        # after_gate = [after_x, after_y, gate_height]
        # waypoints.append(before_gate)
        # waypoints.append(at_gate)
        # waypoints.append(after_gate)
        

        right_x = round(x + gate_half_width * np.cos(yaw), 1)
        right_y = round(y - gate_half_width * np.sin(yaw), 1)
        left_x = round(x - gate_half_width * np.cos(yaw), 1)
        left_y = round(y + gate_half_width * np.sin(yaw), 1)
        right_obs = [right_x, right_y]
        left_obs = [left_x, left_y]
        gate_obs.append(right_obs)
        gate_obs.append(left_obs)

    obst_cent = np.array(cfg['obstacles'])[:, :2]
    obs_bounds = construct_obst_bounds(obst_cent, gate_obs)


    ref_gates = []
    for i in range(len(order)):
        ref_gates.append(Gate(full_gates[order[i]], [order[i]+1]))

    
    waypoints = order_waypoints(startpoint, endpoint, ref_gates, obs_bounds)

    # waypoints.append(endpoint)
    # waypoints = np.array(waypoints)
    # print(waypoints)
    return waypoints, LIFT_HEIGHT, obs_bounds

def construct_obst_bounds(obst_cent, gate_obs):
    gate_obs = np.array(gate_obs)
    crazflie_width = 0.13
    noise = 0.2 + crazflie_width/2 + ADDITIONAL_OBS_BUFFER #obstacle can be moved by this much in x or y

    # Create a new 3D matrix
    obs_bounds = np.zeros((obst_cent.shape[0], 2, 2))

    # Iterate through each element of the original matrix and add/subtract the threshold
    for i in range(obst_cent.shape[0]):
        for j in range(obst_cent.shape[1]):
            obs_bounds[i, 0, j] = obst_cent[i, j] - noise
            obs_bounds[i, 1, j] = obst_cent[i, j] + noise
    gate_obs_bounds = np.zeros((gate_obs.shape[0], 2, 2))
    gate_pillar_width = 0.05 + crazflie_width/2 + GATE_BUFFER
    for i in range(gate_obs.shape[0]):
        for j in range(gate_obs.shape[1]):
            gate_obs_bounds[i, 0, j] = gate_obs[i, j] - gate_pillar_width
            gate_obs_bounds[i, 1, j] = gate_obs[i, j] + gate_pillar_width
    obs_bounds = np.vstack((obs_bounds, gate_obs_bounds))

    return obs_bounds

def does_collide(coord_array, obs_bounds):
    is_colliding = np.any((obs_bounds[:, 0, 0] <= coord_array[0]) & 
                    (coord_array[0] <= obs_bounds[:, 1, 0]) &
                    (obs_bounds[:, 0, 1] <= coord_array[1]) & 
                    (coord_array[1] <= obs_bounds[:, 1, 1]))
    return is_colliding

def RRT_star_solver(start_time, dt, flat_waypoints, cfg, obs_bounds):
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
    # obst_cent = np.array(cfg['obstacles'])[:, :2]
    # obs_bounds = construct_obst_bounds(obst_cent, gate_obs)
    gates_to_pass = int((len(flat_waypoints)-1)/3-1)
    print("gates_to_pass: ", gates_to_pass)

    for i in range(gates_to_pass):
        print("heading to gate ", chr(ord('A') + i))
        if i == 0:
            print("First 6 flat waypoints: ", flat_waypoints[:6])
            total_traj, latest_start =  shortest_curved_path(flat_waypoints[:6], start_time, dt, obs_bounds, last_segment=False)
        else:
            if i == 5:
                DRONE_SPEED = 1.1
            curve, latest_start = shortest_curved_path(flat_waypoints[3*i:3*i+6], latest_start+dt, dt, obs_bounds, last_segment=False)
            total_traj = np.vstack((total_traj, curve))
    print("heading to endpoint")
    last_curve, _ = shortest_curved_path(flat_waypoints[3*(i+1)::], (latest_start+dt), dt, obs_bounds, last_segment=True)
    total_traj = np.vstack((total_traj, last_curve))




    # first_curve, second_start = shortest_curved_path(flat_waypoints[:6], start_time, dt, obs_bounds, last_segment=False)

    # print("To second gate")
    # second_curve, third_start = shortest_curved_path(flat_waypoints[3:9], (second_start+dt), dt, obs_bounds, last_segment=False)

    # print("To third gate")
    # third_curve, fourth_start = shortest_curved_path(flat_waypoints[6:12], (third_start+dt), dt, obs_bounds, last_segment=False)
    
    # print("To fourth gate")
    # fourth_curve, fifth_start = shortest_curved_path(flat_waypoints[9:15], (fourth_start+dt), dt, obs_bounds, last_segment=False)

    # print("To endpoint")
    # print("last waypoints: ", flat_waypoints[12:16])
    # fifth_curve, _            = shortest_curved_path(flat_waypoints[12:16], (fifth_start+dt), dt, obs_bounds, last_segment=True)

    # total_traj = np.vstack((first_curve, second_curve, third_curve, fourth_curve, fifth_curve))


    # value_to_test = np.array([1.4, -2.6])  # 2D value to test
    # collides = does_collide(value_to_test, obs_bounds)
    
    # print(collides)
    return total_traj

def shortest_curved_path(points, time_offset, dt, obs_bounds, last_segment):
    start_node_coord = points[2]
    end_node_coord = points[3]
    print("sending coords: ", start_node_coord, end_node_coord)


    straight_line_traj = RRT_star_straight_lines(start_node_coord, end_node_coord, obs_bounds) #N x 2
    # RRT_len = straight_line_traj.shape[0]
    # print("solution shape: ", straight_line_traj.shape)
    straight_segment_weight = GATE_STRAIGHT_WEIGHT
    straight_seg1 = np.linspace(points[0], points[2], straight_segment_weight)
    
    if last_segment:
        full_traj = np.vstack((straight_seg1, straight_line_traj))
    else:
        straight_seg2 = np.linspace(points[3], points[5], straight_segment_weight)
        full_traj = np.vstack((straight_seg1, straight_line_traj, straight_seg2))
    
    
    
    x_traj = full_traj[:, 0]
    y_traj = full_traj[:, 1]

    degree = POLY_DEGREE
    # if RRT_len < 50:
    #     degree = 4
    coefficients_x = np.polyfit(np.arange(len(full_traj)), x_traj, degree)
    coefficients_y = np.polyfit(np.arange(len(full_traj)), y_traj, degree)

    # Generate the polynomial functions
    poly_function_x = np.poly1d(coefficients_x)
    poly_function_y = np.poly1d(coefficients_y)

    t1 = straight_segment_weight/2
    if last_segment: t2 = len(full_traj)-1
    else: t2 = len(full_traj)-1-(straight_segment_weight/2)
    
      # Example value for t2

    poly_derivative_x = np.polyder(poly_function_x)
    poly_derivative_y = np.polyder(poly_function_y)

    # Define a function to calculate the magnitude of velocity at a given time
    def velocity_magnitude(t):
        vx = poly_derivative_x(t)
        vy = poly_derivative_y(t)
        return np.sqrt(vx**2 + vy**2)


    # Integrate the velocity magnitude function over the time interval
    total_distance, _ = quad(velocity_magnitude, t1, t2)

    print("Distance along the polygonal path:", total_distance, "t2: ", len(full_traj)-1)
    duration = total_distance / DRONE_SPEED
    print("duration: ", duration, "\n\n")

    # Evaluate the polynomial functions to get the fitted x and y values
    intervals = int(duration // dt)
    t = np.linspace(t1, t2, intervals+1)
    fitted_x = poly_function_x(t)
    fitted_y = poly_function_y(t)
    time = np.arange(0, len(fitted_x) * dt, dt)
    time = np.add(time, time_offset)
    last_time = time[-1]

    traj = np.column_stack((time, fitted_x, fitted_y))

    if SHOW_PLOTS:
        plt.plot(x_traj, y_traj, label='Original Trajectory')
        plt.plot(fitted_x, fitted_y, label='Fitted Polynomial Trajectory', linestyle='--')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Trajectory in X-Y Plane')
        plt.legend()
        plt.grid(True)
        plt.xlim(-3.5, 3.5)
        plt.ylim(-3.5, 3.5)
        plt.show()
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
    numer = (end_node_coord[1] - start_node_coord[1])
    denom = (end_node_coord[0] - start_node_coord[0])
    if denom == 0:
        m = 999999
    else: m = numer / denom
    c = start_node_coord[1] - m * start_node_coord[0]
    # print(f"m: {m}, c: {c}")
    std_dev = STD_DEV_SAMPLING

    goal_node = Node((end_node_coord[0], end_node_coord[1]))
    cushion = 0.3
    
    if SHOW_SAMPLES_PLOTS:
        fig, ax = plt.subplots()
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        scatter = ax.scatter([], [])
        plt.ion()
        plt.show()
        x_values = []
        y_values = []

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
        elif abs(m) < 1:
            # Sample a single point along the line using Gaussian distribution
            if start_node_coord[0] < end_node_coord[0]:
                x2 = np.random.uniform(start_node_coord[0]-cushion, end_node_coord[0]+cushion)
            else:
                x2 = np.random.uniform(start_node_coord[0]+cushion, end_node_coord[0]-cushion)
            y2 = m * x2 + c + np.random.normal(0, std_dev)
        else:
            if start_node_coord[1] < end_node_coord[1]:
                y2 = np.random.uniform(start_node_coord[1]-cushion, end_node_coord[1]+cushion)
            else:
                y2 = np.random.uniform(start_node_coord[1]+cushion, end_node_coord[1]-cushion)
            # y2 = m * x2 + c + np.random.normal(0, std_dev)
            x2 = (y2 - c) / m + np.random.normal(0, std_dev)
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

        if SHOW_SAMPLES_PLOTS:
            x_values.append(samp_x)
            y_values.append(samp_y)
            scatter.set_offsets(np.column_stack((x_values, y_values)))
            plt.draw()

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
                # cost_difference = curr_cost - new_cost
                # if 0 < cost_difference < 0.05: #if lower cost with rewiring FIX !!
                if new_cost < curr_cost:
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
    if SHOW_SAMPLES_PLOTS:
        plt.ioff()
        plt.show()
    if SHOW_PLOTS:
        for node in node_list:
            if not node.parent is None:
                xx = [node.x, node.parent.x]
                yy = [node.y, node.parent.y]
                plt.plot(xx, yy, 'k')
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
        coords_x = np.array(coords_x).reshape(-1,1)
        coords_y = np.array(coords_y).reshape(-1,1)
        # coords = np.column_stack((coords_x, coords_y))
        coords = np.hstack((coords_x, coords_y))
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        coords = coords[np.sort(unique_indices)]
        print("start: ", coords[0])
        print("end: ", coords[-1])
        # print("\n\n")

        if SHOW_PLOTS:
            # Plot obstacles
            for obstacle in obs_bounds:
                bottom_left = obstacle[0]
                top_right = obstacle[1]
                width = top_right[0] - bottom_left[0]
                height = top_right[1] - bottom_left[1]
                rectangle = plt.Rectangle((bottom_left[0], bottom_left[1]), width, height, edgecolor='b', facecolor='none')
                plt.gca().add_patch(rectangle)

            # Plot scatter points
            # plt.scatter([point[0] for point in coords], [point[1] for point in coords], color='blue', linestyle='dotted')
            plt.plot([point[0] for point in coords], [point[1] for point in coords], 'ro-', label='Original Coordinates')
            
            # Set plot limits
            plt.xlim(-3.5, 3.5)
            plt.ylim(-3.5, 3.5)
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
    if SHOW_PLOTS:
        plt.grid(True)
        plt.xlim(-3.5, 3.5)
        plt.ylim(-3.5, 3.5)
        plt.show()
    print("No route found with given max_iter, consider increasing")
    return None


def determine_trajectory(csv_path, yaml_path):
    waypoints, _, obs_bounds = load_waypoints(yaml_path)
    print("waypoints: ", waypoints)
    flat_waypoints = waypoints[1:, :2]
    cfg = extract_yaml(yaml_path)
    ctrl_freq = cfg['ctrl_freq']
    dt = 1.0/float(ctrl_freq)
    print("waypoints: \n", waypoints, " \nfirst 3 sent to to_first_waypoints")
    start_3d_traj = to_first_waypoint(dt, waypoints[0:3])
    print("rows of first traj: ", len(start_3d_traj))
    last_time_p1 = start_3d_traj[-1, 0] #latest_time
    start_time = last_time_p1 + dt
    print("\n\n\n")

    middle_2d_traj = RRT_star_solver(start_time, dt, flat_waypoints, cfg, obs_bounds)
    last_time_p2 = middle_2d_traj[-1, 0]
    start_time = last_time_p2 + dt
    # end_3d_traj = to_ground(start_time, dt, waypoints[-2:])

    
    height_column = np.ones((middle_2d_traj.shape[0], 1), dtype=float)
    middle_2d_traj = np.hstack((middle_2d_traj, height_column))

    full_traj = np.vstack((start_3d_traj, middle_2d_traj))
    write_arrays_to_csv(csv_path, full_traj)


if __name__ == '__main__':
    csv_path_sim = 'trajectory.csv'
    yaml_path_sim = 'getting_started.yaml'
    determine_trajectory(csv_path_sim, yaml_path_sim)
