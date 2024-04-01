To run the code for AER1216 lab 4, please run 'lab4_main.py' main method.


Running this file will go through the stereo images, run key point detection and matching.
Using the stereo camera model, the code assigns a 3d position for each of these key points.
We then use pose estimation method (SVD with 10 iterations) to estimate the pose based on these key points.
The code also uses ransac to reject the outliers.

This code has two additions to the starter code provided in the assignment handout:

- Pose Estimation:
    Using two sets of point clouds, for the current frame and previous frame, we used point cloud adjustment to determine
    the change in pose. For point cloud adjustment, we used singular value decomposition, with iteration 5.

- RANSAC:
    to reject the outliers, we used a ransac method by determining the pose change using randomly selected three key points,
    then rejecting the outliers based on that pose estimate. And repeating. We used a Euclidean distance threshold of 0.35 meters
    to reject the outliers. Moreover, the termination condition for ransac was reaching inlier percentage of 95%, or reaching
    11 iterations, which corresponds to 99% confidence

Both of these are added as helper methods in the stereo_vo_base.py file, and used on the visual odometry update function

The method pose_estimation uses ransac on the correlated keypoints 3d points. Each ransac iteration uses the point cloud adjustment method
on 3 random points, then performs outlier rejection. And finally, the point cloud adjustments are used one last time to get the pose estimation
using all the inliers. The SVD is used with 2 iterations during each ransac iteration, and is used with 5 iterations on the final pose estimation