"""
2021-02 -- Wenda Zhao, Miller Tang

This is the code base for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217.
The Kitti dataset, raw_data/City/2011_09_26_drive_0005, is used in this assignment. 
http://www.cvlibs.net/datasets/kitti/raw_data.php
"""

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Kitti Data cannot be saved on Github, too large (340 MB)
#run the following command to download data locally
#   `./downloader.sh`
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ANDRES RANSAC BRANCH

import numpy as np 
import cv2 as cv
import os
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from stereo_vo_base import StereoCamera, VisualOdometry

def main():
    # current working directory
    cwd = os.getcwd()

    # the ground_truth_pose.mat is saved from the kitti 'raw_data_development_kit'->run_demoVehicelPath.m
    # see http://www.cvlibs.net/datasets/kitti/raw_data.php for more details
    OXTS_pose_gt = sio.loadmat('ground_truth_pose.mat')["pose"]

    # image sequence
    sequence_num = OXTS_pose_gt.shape[1]
    pose_gt = np.zeros((sequence_num, 4, 4))
    # save ground truth into pose_gt
    for i in range(sequence_num):
        pose_gt[i] = OXTS_pose_gt[0,i]

    # ---------------------------- Parameter ---------------------------- #
    path_0 = '/CityData/2011_09_26/2011_09_26_drive_0005_sync/image_00/data/'
    path_1 = '/CityData/2011_09_26/2011_09_26_drive_0005_sync/image_01/data/'
    zero_num = 10
    # '/CityData/2011_09_26_calib/calib_cam_to_cam.txt'
                    # baseline, focalLength,   fx,        fy,       cu,       cv
    cam = StereoCamera(0.537,    721.5377, 721.5377, 721.5377, 609.5593, 172.8540)
    vo = VisualOdometry(cam)

    # global transform to camera (starts as identity)
    T = np.eye(4)
    T_hist = np.zeros((sequence_num, 4, 4))
    T_hist[0] = T

    # convert to vehicel frame
    # calibration results provided by kitti dataset in '/CityData/2011_09_26_calib/'
    T_imu_to_velo = np.array([[9.999976e-01,  7.553071e-04,   -2.035826e-03,   -8.086759e-01],
                            [-7.854027e-04,  9.998898e-01,   -1.482298e-02,   3.195559e-01],
                            [2.024406e-03,  1.482454e-02,    9.998881e-01,    -7.997231e-01],
                            [         0,             0,               0,               1] 
                            ])

    T_velo_to_cam = np.array([ [7.533745e-03, -9.999714e-01,  -6.166020e-04,  -4.069766e-03 ],
                            [1.480249e-02, 7.280733e-04,   -9.998902e-01,  -7.631618e-02 ],
                            [9.998621e-01, 7.523790e-03,    1.480755e-02,  -2.717806e-01 ],
                            [           0,            0,               0,              1 ]    
                            ])

    T_cam_to_cam_center = np.array([ [1.0,   0.0,   0.0,  -0.537/2.0],
                                     [0.0,   1.0,   0.0,         0.0],
                                     [0.0,   0.0,   1.0,         0.0],
                                     [0.0,   0.0,   0.0,         1.0]   
                                  ])


    T_imu_to_cam = T_velo_to_cam.dot(T_imu_to_velo)
    
    T_imu_to_cam_center = T_cam_to_cam_center.dot(T_imu_to_cam)

    T_cam_center_to_imu = np.linalg.inv(T_imu_to_cam_center)
    
    # initialization offset: The ground truth data of the vehicle starts with (0, 0, 0) in GPS/IMU frame. But in the VO estimation, we set initial position of the stereo-camera's center as (0,0,0) using initial r. Therefore, we convert the ground truth data to camera center position in GPS frame with (0, 0, 0) as the initial position.  The convertion is done by translation vector t_cv_v.

    # translation from vehicle frame to camera center frame expressed in vehicle frame (from Kitti website)
    t_cv_v = np.array([1.09, -0.32-0.537/2.0, 0.8])

    cam_center_gt = np.zeros((sequence_num, 3))
    for idx in range(sequence_num):
        cam_center_gt[idx,:] = pose_gt[idx,0:3,3]  + t_cv_v
    
    # Transformation in vehicle frame
    T_vehicle = np.zeros((sequence_num, 4, 4))

    # record video
    os.chdir(cwd)
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    video = cv.VideoWriter('./video.avi', fourcc, 5.0, (1242, 775))  # 375*2 + 25 (margin)

    for img_id in range(sequence_num):
        img_left  = cv.imread(cwd + path_0 + str(img_id).zfill(zero_num) + '.png', 0)
        img_right = cv.imread(cwd + path_1 + str(img_id).zfill(zero_num)+'.png', 0)

        # finite-state machine
        # update the C and r 
        frame_left, frame_right = vo.update(img_left, img_right, img_id)
        
        # Create a white margin between two frames
        margin = np.ones_like(frame_left)*255
        vertical_frame = np.concatenate((frame_left, margin[0:25], frame_right), axis=0) 
        
        # Update the transformation matrix
        T_update = np.vstack( ((np.concatenate((vo.C, (vo.r).reshape(-1,1)), axis = 1)),
                                np.array([[0,0,0,1]]) ))
        T = T_update.dot(T)
        # Store the history of transformation matrix
        T_hist[img_id]=T
        # convert to vehicle frame
        T_vehicle[img_id] = T_cam_center_to_imu.dot(np.linalg.inv(T_hist[img_id]))
        
        cv.imshow('Visual Odometry', vertical_frame)
        video.write(vertical_frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
        
    print("VO ends\n")
    video.release()
    cv.destroyAllWindows()
    
    # save the estimated transformation matrix 
    np.save('VO_T.npy', T_vehicle)

    fig_traj = plt.figure(facecolor = "white")
    ax_t = fig_traj.add_subplot(111, projection = '3d')
    ax_t.plot(cam_center_gt[:,0], cam_center_gt[:,1], cam_center_gt[:,2], color='red',label='ground truth trajectory', linewidth=1.9, alpha=0.9)
    ax_t.plot(T_vehicle[:,0,3],T_vehicle[:,1,3],T_vehicle[:,2,3],color='steelblue',label='VO trajectory',linewidth=1.9, alpha=0.9)
    ax_t.set_xlabel(r'X [m]')
    ax_t.set_ylabel(r'Y [m]')
    ax_t.set_zlabel(r'Z [m]')
    ax_t.legend()
    ax_t.set_zlim3d(-20.0, 25.0)
    plt.title(r"Trajectory of the vehicle", fontsize=13, fontweight=0, color='black', style='italic', y=1.02 )
    plt.show()

if __name__ == '__main__':
    print('We are using OpenCV version {}'.format(cv.__version__))
    main()
