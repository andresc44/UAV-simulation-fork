#placeholder for Lab3 source code
import numpy as np
import cv2
import pandas as pd
import os
#data in lab2_pose.csv and image_folder/output_folder/*

#6 targets
#The targets, as shown in Figure 1 will be within an area on the ground bounded by x ∈ [−2.0, 2.0] m and y ∈ [−2.0, 2.0] m.
#640px × 360px)
#six targets 
#Hint: You should use more than one image per target to improve your estimation.
current_directory = os.getcwd()
file_name = 'lab3_pose.csv'
file_path = os.path.join(current_directory, file_name)


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
    print(df.head())
    



if __name__ == '__main__':
    main()