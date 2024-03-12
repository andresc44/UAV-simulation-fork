#placeholder for Lab3 source code
import numpy as np
import cv2
import pandas as pd
import os


from scipy.spatial.transform import Rotation
#data in lab2_pose.csv and image_folder/output_folder/*

#6 targets
#The targets, as shown in Figure 1 will be within an area on the ground bounded by x ∈ [−2.0, 2.0] m and y ∈ [−2.0, 2.0] m.
#640px × 360px)
#six targets 
#Hint: You should use more than one image per target to improve your estimation.

ITERATIONS = 20

np.random.seed(1217)
current_directory = os.getcwd()
file_name = 'lab3_pose.csv'
file_path = os.path.join(current_directory, file_name)
image_dir = 'image_folder/output_folder'
image_dir = os.path.join(current_directory, image_dir)
image_jpgs = os.listdir(image_dir)


def get_target_location(image_file_path, K, d):
    image_path = os.path.join(image_dir, image_file_path)
    image = cv2.imread(image_path)
    if image is None:
        print("No image")
        return None
    
    #instrisic K
    #distortion coefficents d
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    undistorted_image = cv2.undistort(image, K, d) #Get corrected image
    
    cv2.imshow('undistorted', undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hsv = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for green color in HSV
    lower_green = np.array([30, 80, 40])
    upper_green = np.array([80, 255, 255])

    masked_image = cv2.inRange(hsv, lower_green, upper_green)

    # cv2.imshow('masked', masked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours in the masked_image
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = cv2.drawContours(masked_image, contours, -1, (0, 255, 0), 3)

    # cv2.imshow('Contours', contour_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    min_area= 100
    max_area = 800

    # Check if any contours are found
    if len(contours) > 0:
        # Get the largest contour (assuming it corresponds to the circle)
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the moments of the contour to compute the centroid
        # print("Area size: ", cv2.contourArea(largest_contour))
        if min_area < cv2.contourArea(largest_contour) < max_area: #Very small contour
        
            M = cv2.moments(largest_contour)
            # Calculate centroid coordinates
            centroidX = int(M["m10"] / M["m00"])
            centroidY = int(M["m01"] / M["m00"])
            # print("centroid X", centroidX, "centroid Y: ", centroidY)

            c_x = K[0][2]
            c_y = K[1][2]
            f_x = K[0][0]
            f_y = K[1][1]

            Z = 1 #need real depth
            X = Z * (centroidX - c_x) / f_x
            Y = Z * (centroidY - c_y) / f_y

            target_cam_coordinates = np.array([[X], [Y], [Z], [1]])
            return target_cam_coordinates # return target_cam_coordinates #4x1 [[x], [y], [z], 1]
    return None

def tf_cam2vehicle(target_cam_coordinates, T_CB): #target_cam_coordinates is 4x1
    T_BC  = np.linalg.inv(T_CB) #4x4
    target_vehicle_coordinates = np.matmul(T_BC, target_cam_coordinates)
    return target_vehicle_coordinates #4x1

def tf_vehicle2world(target_vehicle_coordinates, df, i):
    row = df[df['idx'] == i]
    row_np = row.values[0]
    # print("row_np: ", row_np)
    vehicle_translations = row_np[1:4].T
    vehicle_quaternion = row_np[4:]
    
    # print("vehicle_translations: ", vehicle_translations)
    # print("vehicle_quaternion: ", vehicle_quaternion)

    rotation_matrix = Rotation.from_quat(vehicle_quaternion).as_matrix()
    T_BW = np.eye(4)  # Identity matrix
    T_BW[:3, :3] = rotation_matrix  # Set rotation values
    T_BW[:3, 3] = vehicle_translations  # Set translation values
    T_WB  = np.linalg.inv(T_BW)
    points_world = np.matmul(T_WB, target_vehicle_coordinates)
    return points_world #4x1

def within_bounds(target_world_coordinates):
    x = target_world_coordinates[0][0]
    y = target_world_coordinates[1][0]
    x_constrained = min(max(x, -2), 2)
    y_constrained = min(max(y, -2), 2)
    return np.array([[x_constrained, y_constrained]])

def get_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    squared_diffs = (x2 - x1) ** 2 + (y2 - y1) ** 2
    distance = np.sqrt(squared_diffs)
    return distance

def assign_cluster(cluster_centres, valid_world_targets): #6x2, ?x2
    min_dist = float('inf')
    cluster_ids = []
    for point in valid_world_targets:
        for id, cluster_point in enumerate(cluster_centres):
            distance = get_distance(point, cluster_point)
            if distance < min_dist:
                min_dist = distance
                cluster_id = id
        cluster_ids.append(cluster_id)
    cluster_ids = np.array(cluster_ids)
    return cluster_ids

def recentre_clusters(cluster_ids, k_clusters, valid_world_targets, cluster_centres): #?, ?x2
    for id in range(k_clusters):
        mask = (cluster_ids == id)
        members = valid_world_targets[mask]
        num_members = len(members)
        if num_members == 0:
            continue
        x_sum = 0
        y_sum = 0
        for point in members:
            x_sum += point[0]
            y_sum += point[1]
        x_centre = x_sum / num_members
        y_centre = y_sum / num_members
        cluster_centres[id] = [x_centre, y_centre]
    cluster_centres = np.array(cluster_centres)

    return cluster_centres



def main():
    K = np.array([              #Camera Intrinsic Matrix
        [698.86, 0.0, 306.91],
        [0.0, 699.13, 150.34],
        [0.0, 0.0, 1.0]
        ])
    d = np.array([0.191887, -0.563680, -0.003676, -0.002037, 0.0]) #Distortion Coefficients
    T_CB = np.array([ #extrinsic transformation matrix from the vehicle body (Vicon) frame to the camera frame
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
        ])
    df = pd.read_csv(file_path) #Vicon interial vehicle information,  x ∈ [−2.0, 2.0] m and y ∈ [−2.0, 2.0] m.

    valid_world_targets = []
    for i, image_file_path in enumerate(image_jpgs):
        if i == 20:
            break
        depth = calculate_depth_from_camera(df, T_CB)#?????
        
        target_cam_coordinates = get_target_location(image_file_path, K, d) #return 4x1
        # print("target_cam_coordinates: ", target_cam_coordinates)
        if isinstance(target_cam_coordinates, np.ndarray): #valid point found
            print("VALID IMAGE")
            target_vehicle_coordinates = tf_cam2vehicle(target_cam_coordinates, T_CB)
            # print("target_vehicle_coordinates: ", target_vehicle_coordinates)
            target_world_coordinates = tf_vehicle2world(target_vehicle_coordinates, df, i)
            # print("target_world_coordinates: ", target_world_coordinates)
            corrected_target_world_coordinates = within_bounds(target_world_coordinates)
            # print("corrected_target_world_coordinates: ", corrected_target_world_coordinates)

            if i==0:
                valid_world_targets = corrected_target_world_coordinates
            else:
                valid_world_targets = np.vstack((valid_world_targets, corrected_target_world_coordinates))
    
    valid_world_targets = np.array(valid_world_targets)
    print("\nvalid_world_targets: ", valid_world_targets)

    k_clusters = 6
    cluster_centres = np.random.uniform(low=-2, high=2, size=(k_clusters, 2))
    # print("random cluster centres: ", cluster_centres, "\n")
    cluster_ids = assign_cluster(cluster_centres, valid_world_targets) #1D array of length = len(valid_world_targets)

    for _ in range(ITERATIONS):
        cluster_centres = recentre_clusters(cluster_ids, k_clusters, valid_world_targets, cluster_centres)
        cluster_ids = assign_cluster(cluster_centres, valid_world_targets)
    
    target_estimates = recentre_clusters(cluster_ids, k_clusters, valid_world_targets, cluster_centres)
        
    print("6 targets estimated at: ")
    print(target_estimates)
    return target_estimates



if __name__ == '__main__':
    main()