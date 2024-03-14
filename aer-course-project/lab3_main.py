#placeholder for Lab3 source code
import numpy as np
import cv2
import pandas as pd
import os
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

#TODO: -verify tf_vehicle2world !!!!
    #  -improve the way depth is determined by including pixel location information (low priority)
    #  -check for edge cases for image pipeline
    #  -after any major changes, check for clustering in Kmeans chart


#data in lab2_pose.csv and image_folder/output_folder/*

#6 targets
#The targets, as shown in Figure 1 will be within an area on the ground bounded by x ∈ [−2.0, 2.0] m and y ∈ [−2.0, 2.0] m.
#640px × 360px)
#six targets 
#Hint: You should use more than one image per target to improve your estimation.

ITERATIONS = 50
TOLERANCE = 0.3
np.random.seed(1217)
current_directory = os.getcwd()
file_name = 'lab3_pose.csv'
file_path = os.path.join(current_directory, file_name)
image_dir = 'image_folder/output_folder'
image_dir = os.path.join(current_directory, image_dir)
image_jpgs = sorted(os.listdir(image_dir), key=None)

def visualize_transform(T):
    # Extract rotation matrix and translation vector from the transformation matrix
    R = T[:3, :3]
    t = T[:3, 3]

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define a function to plot a coordinate frame
    def plot_frame(ax, R, t, label):
        ax.quiver(t[0], t[1], t[2], R[0, 0], R[1, 0], R[2, 0], color='r', length=0.1, arrow_length_ratio=0.3)
        ax.quiver(t[0], t[1], t[2], R[0, 1], R[1, 1], R[2, 1], color='g', length=0.1, arrow_length_ratio=0.3)
        ax.quiver(t[0], t[1], t[2], R[0, 2], R[1, 2], R[2, 2], color='b', length=0.1, arrow_length_ratio=0.3)
        ax.text(t[0], t[1], t[2], label, color='k')

    # Plot coordinate frame
    plot_frame(ax, R, t, 'Transform')

    # Set plot limits and labels
    buffer = 0.2
    ax.set_xlim([t[0] - buffer, t[0] + buffer])
    ax.set_ylim([t[1] - buffer, t[1] + buffer])
    ax.set_zlim([t[2] - buffer, t[2] + buffer])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()



########## DEPTH ###############################################################
def calculate_depth_from_camera(df, i):
    row = df[df['idx'] == i]
    if row.empty:
        raise ValueError("Index not found in DataFrame")
    
    row_np = row.values[0]
    # print("\n\ni: ", i)
    vehicle_translations = row_np[1:4]
    # print("translations: ", vehicle_translations)
    vehicle_quaternion = np.append(row_np[5:8],row_np[4]) #xyzw
    # print("vehicle_quaternion: ",vehicle_quaternion)
    
    R_BW = Rotation.from_quat(vehicle_quaternion).as_matrix() #correct, orthogonal, det(R)==1 SO group
    # R_CB = T_CB[:3, :3]
    # print("R_CB", R_CB)
    # det = np.linalg.det(R_CB)

    # print("Determinant of the matrix: ")
    # print(det)
    # print(np.matmul(R_CB, R_CB.T))
    # print("Rotation: ", R_BW)
    # print("translation: ", vehicle_translations)
    # T_BW = np.eye(4)  # Identity matrix
    # T_BW[:3, :3] = R_BW  # Set rotation values
    # T_BW[:3, 3] = vehicle_translations  # Set translation values
    
    #T_BW is part of SE group, where its rotation is orthogonal and a pure rotation
    #transformation from world to body frame
    # print("R_BW: \n", R_BW)
    # print("T_BW: \n", T_BW)
    # visualize_transform(T_BW)

    #T_BW is formed, transform from world to base, 4x4
    # T_WB = np.linalg.inv(T_BW) #transformation from body to world frame
    # T_CW = np.dot(T_BW,T_CB) #correct, verified, part of SE
    # print("original: ", T_BW)
    # print("inverted: ", T_WB)
    # print("transposed: ", T_BW.T)
    # is_orthogonal = np.allclose(np.dot(R_BW, R_BW.T), np.eye(3))
    # print("Is the matrix orthogonal? ", is_orthogonal)

    # print("should be identity: ", np.matmul(T_BW.T, T_BW))
    # print("T_CB (base to camera): \n", T_CB)
    # print("T_BW (world to base): \n", T_BW)
    # print("T_CW: \n", T_CW)
    # visualize_transform(T_BW)
    # visualize_transform(T_CW)
    # T_WC = np.linalg.inv(T_CW)
    # visualize_transform(T_WC)
    # print("T_WC (camera to world): \n", T_WC)
    
    # T_BC = np.linalg.inv(T_CB)
    # camera_origin = np.array([0, 0, 0, 1]).T #location of camera, 4x1
    # cam_coords_in_world = np.dot(np.dot(T_BC, T_WB), camera_origin) #global location of camera, 4x1
    # print(cam_coords_in_world)
    # cam_height = cam_coords_in_world[2]
    cam_height = vehicle_translations[2]
    angle_difference = np.arccos(np.dot(R_BW[:, 2], np.array([0, 0, 1])))
    angle_difference_degrees = np.degrees(angle_difference)

    depth = cam_height / math.cos(angle_difference) #trigonometry. Approximated based off height and tilt, not pixel found
    # print("tilt: ", angle_difference_degrees, "\tcam height: ", cam_height, "m\t approx depth: ", depth)
    return depth, R_BW, vehicle_translations #would be depth, but cant trust rotation information


############ IMAGE PROCESSING ####################################################
def get_target_location(image_file_path, K, d, depth):
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
    
    # cv2.imshow('undistorted', undistorted_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hsv = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for green color in HSV
    lower_green = np.array([35, 70, 20])
    upper_green = np.array([80, 255, 255])

    masked_image = cv2.inRange(hsv, lower_green, upper_green)

    # cv2.imshow('masked', masked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours in the masked_image
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    min_area= 100
    max_area = 700 #exclude green tape
    min_aspect_ratio = 0.5  # Adjust as necessary
    max_aspect_ratio = 2.0  # Adjust as necessary

    # Check if any contours are found
    if len(contours) > 0:
        valid_cnt = []
        for cnt in contours:
            if min_area < cv2.contourArea(cnt) < max_area:
                _, _, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                    valid_cnt.append(cnt)
        
        
        if len(valid_cnt) > 0: #Very small contour  
            # Get the largest contour (assuming it corresponds to the circle)
            largest_contour = max(valid_cnt, key=cv2.contourArea)
            # Get the moments of the contour to compute the centroid
            # print("Area size: ", cv2.contourArea(largest_contour))
            # cv2.drawContours(undistorted_image, [largest_contour], -1, (255, 0, 255), 3)
            # cv2.imshow('Contours', undistorted_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            M = cv2.moments(largest_contour)
            
            # Calculate centroid coordinates
            centroidX = int(M["m10"] / M["m00"])
            centroidY = int(M["m01"] / M["m00"])
            # print("centroid X", centroidX, "centroid Y: ", centroidY)

            c_x = K[0][2]
            c_y = K[1][2]
            f_x = K[0][0]
            f_y = K[1][1]

            Z = depth #need real depth
            X = Z * (centroidX - c_x) / f_x
            Y = Z * (centroidY - c_y) / f_y

            target_cam_coordinates = np.array([[X], [Y], [Z], [1]])
            return target_cam_coordinates # return target_cam_coordinates #4x1 [[x], [y], [z], 1]
    return None



########## TRANSFORMS ############################################################

def tf_cam2vehicle(target_cam_coordinates, T_CB): #target_cam_coordinates is 4x1
    T_BC  = np.linalg.inv(T_CB) #4x4, is equal to its inverse
    target_vehicle_coordinates = np.matmul(T_BC, target_cam_coordinates)
    return target_vehicle_coordinates #4x1

   
def tf_vehicle2world(target_vehicle_coordinates, R_BW, vehicle_translations):
    #target_vehicle_coordinates is 4x1, R_BW is 3x3 from map to vehicle frame, vehicle_translations is len 3
    #use visualize_transform fxn to see full transformations

    R_WB = np.linalg.inv(R_BW) #rotate in opposite direction
    rot_T_WB = np.eye(4)
    # rot_T_WB[:3, :3] = R_WB #affine pure rot matrix
    rot_T_WB[:3, :3] = R_BW #affine pure rot matrix

    # visualize_transform(rot_T_WB)
    # print("rot_T_WB: ", rot_T_WB)
    
    vehicle_translations= np.append(vehicle_translations, 0)
    vehicle_translations = vehicle_translations.reshape(4, 1)

    points_rotated = np.dot(rot_T_WB, target_vehicle_coordinates)
    # print("points_rotated: ", points_rotated)
    # print("vehicle_translations: ", vehicle_translations)
    points_world = vehicle_translations + points_rotated
    return points_world #4x1

def within_bounds(target_world_coordinates):
    x = target_world_coordinates[0][0]
    y = target_world_coordinates[1][0]
    x_constrained = min(max(x, -2), 2)
    y_constrained = min(max(y, -2), 2)
    return np.array([[x_constrained, y_constrained]])

####### K-MEANS CLUSTERING##################################################
def get_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    squared_diffs = (x2 - x1) ** 2 + (y2 - y1) ** 2
    distance = np.sqrt(squared_diffs)
    return distance

def assign_cluster(cluster_centres, valid_world_targets): #6x2, ?x2
    
    cluster_ids = []
    for point in valid_world_targets:
        min_dist = float('inf')
        for c_id, cluster_point in enumerate(cluster_centres):
            distance = get_distance(point, cluster_point)
            if distance < min_dist:
                min_dist = distance
                cluster_id = c_id
        cluster_ids.append(cluster_id)
    cluster_ids = np.array(cluster_ids)
    return cluster_ids

def recentre_clusters(cluster_ids, k_clusters, valid_world_targets, cluster_centres):
    for c_id in range(k_clusters):
        mask = (cluster_ids == c_id)
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
        cluster_centres[c_id] = [x_centre, y_centre]
    cluster_centres = np.array(cluster_centres)

    return cluster_centres

def filter_coordinates(cluster_centres, datapoints, tol):
    filtered_coords = []
    for coord in datapoints:
        for centre in cluster_centres:
            distance = get_distance(centre, coord)
            if distance <= tol:
                filtered_coords.append(coord)
                break 
    clean_targets = np.array(filtered_coords)
    return clean_targets

def mini_Kmeans(k_clusters, datapoints, cluster_centres):
    cluster_ids = assign_cluster(cluster_centres, datapoints) #1D array of length = len(datapoints)
    for _ in range(ITERATIONS):
        cluster_centres = recentre_clusters(cluster_ids, k_clusters, datapoints, cluster_centres)
        cluster_ids = assign_cluster(cluster_centres, datapoints)
    estimated_final_centres = recentre_clusters(cluster_ids, k_clusters, datapoints, cluster_centres)
    return estimated_final_centres

def plot_Kmeans_results(datapoints, clusters):
    
    plt.figure(figsize=(8, 6))
    plt.scatter(datapoints[:, 0], datapoints[:, 1], color='blue', label='Datapoints')
    plt.scatter(clusters[:, 0], clusters[:, 1], color='red', label='Clusters')
    plt.title('Overlay of Scatter Plots')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def apply_full_KMeans(valid_world_targets):
    k_clusters = 6
    cluster_centres = np.array([
        [-1.0, 1.0],
        [0.0, -1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.5, -1.0]
    ])
    ordinary_centres = mini_Kmeans(k_clusters, valid_world_targets, cluster_centres)
    plot_Kmeans_results(valid_world_targets, ordinary_centres)
    
    clean_targets = filter_coordinates(ordinary_centres, valid_world_targets, TOLERANCE)
    clean_centres = mini_Kmeans(k_clusters, clean_targets, ordinary_centres)
    plot_Kmeans_results(clean_targets, clean_centres)

    ultra_clean_targets = filter_coordinates(clean_centres, clean_targets, 0.1)
    ultra_clean_centres = mini_Kmeans(k_clusters, ultra_clean_targets, clean_centres)
    plot_Kmeans_results(ultra_clean_targets, ultra_clean_centres)

    return ultra_clean_centres


############## MAIN ################################################################
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
    no_valid_images_found = True
    valid_world_targets = []
    for image_file_path in image_jpgs:
        i = int(image_file_path[6:-4])
        print("i: ", i)
        # if i < 1200: #just skip to middle images for debugging
        #     continue
        depth, R_BW, vehicle_translations = calculate_depth_from_camera(df, i)
        
        target_cam_coordinates = get_target_location(image_file_path, K, d, depth) #return 4x1
        # print("target_cam_coordinates: ", target_cam_coordinates)
        if isinstance(target_cam_coordinates, np.ndarray): #valid point found
            print("VALID IMAGE")
            target_vehicle_coordinates = tf_cam2vehicle(target_cam_coordinates, T_CB)
            # print("target_vehicle_coordinates: \n", target_vehicle_coordinates)
            target_world_coordinates = tf_vehicle2world(target_vehicle_coordinates, R_BW, vehicle_translations)
            # print("target_world_coordinates: \n", target_world_coordinates)
            corrected_target_world_coordinates = within_bounds(target_world_coordinates)
            # print("corrected_target_world_coordinates: \n", corrected_target_world_coordinates)

            if no_valid_images_found:
                valid_world_targets = corrected_target_world_coordinates
                no_valid_images_found = False
            else:
                valid_world_targets = np.vstack((valid_world_targets, corrected_target_world_coordinates))
        else:
            print("INVALID IMAGE")
    
    valid_world_targets = np.array(valid_world_targets)
    final_clusters_prediction = apply_full_KMeans(valid_world_targets)
    print("6 targets estimated at: ")
    print(final_clusters_prediction)
    return final_clusters_prediction

if __name__ == '__main__':
    target_estimates = main()