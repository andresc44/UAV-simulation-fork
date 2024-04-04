"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a stereo visual Odometry designed
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""

import math

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

# np.random.rand(1217) In starter code, probably typo
np.random.seed(1217)


class StereoCamera:
    def __init__(self, baseline, focal_length, fx, fy, cx, cy):
        self.baseline = baseline
        self.f_len = focal_length
        self.fx = fx
        self.fy = fy
        self.cu = cx
        self.cv = cy


def find_feature_correspondences(
    kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r
):
    vertical_px_buffer = 1  # buffer for the epi-polar constraint in number of pixels
    far_thresh = 7  # 7 pixels are approximately 55m away from the camera
    close_thresh = 65  # 65 pixels are approximately 4.2m away from the camera

    number_of_features = len(kp_l)
    bf = cv.BFMatcher(
        cv.NORM_L2, crossCheck=True
    )  # BFMatcher for SIFT or SURF features matching

    # using the current left image as the anchor image
    match_l_r = bf.match(des_l, des_r)  # current left to current right
    match_l_l_prev = bf.match(des_l, des_l_prev)  # cur left to prev. left
    match_l_r_prev = bf.match(des_l, des_r_prev)  # cur left to prev. right

    kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
    kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
    kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

    kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
    kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
    kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

    # loop through all the matched features to find common features
    features_corr = np.zeros((1, 8))
    for pt_idx in np.arange(number_of_features):
        if (
            (pt_idx in set(kp_query_idx_l_r))
            and (pt_idx in set(kp_query_idx_l_l_prev))
            and (pt_idx in set(kp_query_idx_l_r_prev))
        ):
            temp_feature = np.zeros((1, 8))
            temp_feature[:, 0:2] = kp_l_prev[
                kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]
            ].pt
            temp_feature[:, 2:4] = kp_r_prev[
                kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]
            ].pt
            temp_feature[:, 4:6] = kp_l[pt_idx].pt
            temp_feature[:, 6:8] = kp_r[
                kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]
            ].pt
            features_corr = np.vstack((features_corr, temp_feature))
    features_corr = np.delete(features_corr, 0, axis=0)

    #  additional filter to refine feature correspondences
    # 1. drop those features do NOT follow the epi-polar constraint
    features_corr = features_corr[
        (np.absolute(features_corr[:, 1] - features_corr[:, 3]) < vertical_px_buffer)
        & (np.absolute(features_corr[:, 5] - features_corr[:, 7]) < vertical_px_buffer)
    ]

    # 2. drop those features that are either too close or too far from the cameras
    features_corr = features_corr[
        (np.absolute(features_corr[:, 0] - features_corr[:, 2]) > far_thresh)
        & (np.absolute(features_corr[:, 0] - features_corr[:, 2]) < close_thresh)
    ]

    features_corr = features_corr[
        (np.absolute(features_corr[:, 4] - features_corr[:, 6]) > far_thresh)
        & (np.absolute(features_corr[:, 4] - features_corr[:, 6]) < close_thresh)
    ]
    # features_corr:
    #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
    return features_corr


def estimate_pose(corresponding_pair_source, corresponding_pair_target):

    # 1. Compute the centroid of the point clouds without weights
    source_centroid = np.mean(corresponding_pair_source, axis=0)
    target_centroid = np.mean(corresponding_pair_target, axis=0)
    corresponding_pair_source = corresponding_pair_source - source_centroid
    corresponding_pair_target = corresponding_pair_target - target_centroid
    # 2. Compute the outer product (covariance matrix)
    covariance_matrix = np.sum(
        np.array(
            [
                np.matmul(
                    corresponding_pair_source[i].reshape(3, 1),
                    corresponding_pair_target[i].reshape(1, 3),
                )
                for i in range(len(corresponding_pair_source))
            ]
        ),
        axis=0,
    )

    # 3. Using Singular Value Decomposition (SVD)
    u, s, vt = np.linalg.svd(covariance_matrix)
    v = np.transpose(vt)

    # 4. Compose final rotation and translation
    sign_identity = np.eye(3)
    sign_identity[2, 2] = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    rotation_matrix_target_source = np.matmul(
        v, np.matmul(sign_identity, np.transpose(u))
    )
    translation_vector_target_source_target = (
        -np.matmul(np.transpose(rotation_matrix_target_source), target_centroid)
        + source_centroid
    )
    translation_matrix_target_source = np.identity(4)
    translation_matrix_target_source[0:3, 0:3] = rotation_matrix_target_source
    translation_matrix_target_source[0:3, 3] = -np.matmul(
        rotation_matrix_target_source, translation_vector_target_source_target
    )

    return translation_matrix_target_source


# def nearest_search(source_point_cloud, target_point_cloud):
#     # Using brute force search, we will compute the Euclidean distance between each two pairs, then take the minimum distance for each source point
#     corresponding_pair_source = source_point_cloud
#     corresponding_pair_target = np.zeros(corresponding_pair_source.shape)
#     euclidean_distance_summation = 0
#     for source_index in range(len(source_point_cloud)):
#         min_euclidean_distance = np.linalg.norm(
#             source_point_cloud[source_index] - target_point_cloud[0]
#         )
#         for target_index in range(len(target_point_cloud)):
#             euclidean_distance = np.linalg.norm(
#                 source_point_cloud[source_index] - target_point_cloud[target_index]
#             )
#             if euclidean_distance <= min_euclidean_distance:
#                 corresponding_pair_target[source_index] = target_point_cloud[
#                     target_index
#                 ]
#                 min_euclidean_distance = euclidean_distance
#         euclidean_distance_summation += min_euclidean_distance

#     mean_nearest_euclidean_distance = euclidean_distance_summation / len(
#         source_point_cloud
#     )
#     return (
#         corresponding_pair_source,
#         corresponding_pair_target,
#         mean_nearest_euclidean_distance,
#     )


def point_cloud_alignment_matrix(
    source_point_cloud, target_point_cloud, number_of_iterations=1
):
    """
    ...

    Parameters:
    source_point_cloud   : numpy.ndarray size: (?, 3) either (M x 3) or (3 x 3)
    target_point_cloud    : numpy.ndarray size: (?, 3) either (M x 3) or (3 x 3)
    number_of_iterations    : number of iterations
    ...

    Returns:
    transform_matrix   : numpy.ndarray   size: (4, 4)
    ...
    """
    pose = np.identity(4)
    updated_point_cloud = source_point_cloud
    # iteration_mean_euclidean_distance = [0] * number_of_iterations

    for i in range(number_of_iterations):
        # (
        #     corresponding_pair_source,
        #     corresponding_pair_target,
        #     iteration_mean_euclidean_distance[i],
        # ) = nearest_search(updated_point_cloud, target_point_cloud)
        pose_translation_matrix = estimate_pose(updated_point_cloud, target_point_cloud)
        updated_point_cloud_reshaped = np.vstack(
            [np.transpose(updated_point_cloud), np.ones(len(updated_point_cloud))]
        )
        updated_point_cloud = np.matmul(
            pose_translation_matrix, updated_point_cloud_reshaped
        )
        updated_point_cloud = np.transpose(updated_point_cloud[0:3, :])
        pose = np.matmul(pose_translation_matrix, pose)

    return pose


def feature_tracking(prev_kp, cur_kp, img, color=(0, 255, 0), alpha=0.5):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cover = np.zeros_like(img)
    # Draw the feature tracking
    for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        cover = cv.line(cover, (a, b), (c, d), color, 2)
        cover = cv.circle(cover, (a, b), 3, color, -1)
    frame = cv.addWeighted(cover, alpha, img, 0.75, 0)

    return frame


def compute_max_iter_ransac(certainty):
    """
    Use the RANSAC eqn to determine a max number of iterations to perform to gain a degree of certainty in
    inlier detection. Max_iter = frac{log(1 - p)}{log(1 - (1-e)^s)}, where p is the certainty, e
    is the estimated proportion of outliers, and s is the number of samples needed to estimate the model.

    Parameters:
    certainty : float
    value from 0 to 1 (non-inclusive) of how certain we want to be. Usually set to 0.99 or higher

    Returns:
    int
    The number of iterations required to achieve the desired certainty
    """
    pts_picked = 3
    perc_outlier = 0.3
    number_of_iterations = math.log(1 - certainty) / math.log(
        1 - (1 - perc_outlier) ** pts_picked
    )
    return math.ceil(number_of_iterations)


def calculate_euclid(p_cur, p_transformed):
    """
    Determine the Euclidean distance of each feature between the current frame, and the transformation
    applied to the previous frame.

    Parameters:
    p_cur           : numpy.ndarray   size: (N, 3)
    p_transformed   : numpy.ndarray   size: (N, 3)
    2 supposedly similar frames of features.

    Returns:
    numpy.ndarray size: (N, 1)
    Each element represents the Euclidean distance for that row between the 2 frames
    """
    squared_diff = (p_cur - p_transformed) ** 2
    sum_squared_diff = np.sum(squared_diff, axis=1)
    dist = np.sqrt(sum_squared_diff)
    return dist


def filter_inliers_ransac(p_before, p_cur, max_iter):
    """
    Apply RANSAC inlier filtering to the given data to only preserve the features that are inliers.

    Parameters:
    p_before    : numpy.ndarray   size: (number_of_pairs, 3)
    p_cur       : numpy.ndarray   size: (number_of_pairs, 3)
    max_iter    : int

    The np arrays are each the unfiltered 3D point clouds of the features from the previous and current frame.
    Max_iter is the number of iterations of RANSAC required to achieve a certain confidence level (0.99).

    Returns:
    p_a             : numpy.ndarray   size: (M, 3)
    p_b             : numpy.ndarray   size: (M, 3)
    inlier_indices  : numpy.ndarray   size: (M, 1);

    Info:
    number_of_pairs >= M. The returned arrays are the 3D point clouds of the inlier features for the previous and current
    frames, respectively. The elements of inlier_indices represent the indices of number_of_pairs that were considered inliers.
    """

    max_inliers = -1
    top_t = np.zeros((4, 4))
    inlier_threshold_euclidean = 0.35
    high_inlier_perc_threshold = 0.95  # more than the high_inlier_perc_threshold percent of points are inliers, can break RANSAC
    number_of_pairs = p_before.shape[0]

    def inliers_from_t(transform_matrix, pairs_before, pairs_current):
        """
        Given a certain transform transform_matrix, apply that transform to the previous frame point cloud and determine
        the inliers by comparing the Euclidean distance to the current frame point cloud.

        Parameters:
        transform_matrix           : numpy.ndarray size: (4, 4)
        p_before    : numpy.ndarray size: (number_of_pairs, 3)
        p_cur       : numpy.ndarray size: (number_of_pairs, 3)

        The np arrays are each the unfiltered 3D point clouds of the features from the previous and current frame.

        Returns:
        inliers_cnt     : int           value <= number_of_pairs
        inlier_indices  : numpy.ndarray size: (number_of_inliers, 1), and each element is value from 0-> number_of_pairs-1

        Returns the number of inliers found, as well as the indices of those inliers w.r.t. The original
        number_of_pairs features.
        """
        p_before_transform = np.hstack(
            (pairs_before, np.ones((number_of_pairs, 1)))
        ).T  # 4 x number_of_pairs where the last row is 1s

        p_after_transform = (
            transform_matrix @ p_before_transform
        )  # 4x4 @ 4 x number_of_pairs => 4 x number_of_pairs
        p_transformed = p_after_transform[:3, :].T  # number_of_pairs x 3 X, Y, Z

        dist = calculate_euclid(pairs_current, p_transformed)  # number_of_pairs,
        mask = dist < inlier_threshold_euclidean  # Boolean len number_of_pairs
        this_inlier_indices = np.where(mask)[0]  # M,
        this_number_of_inliers = np.count_nonzero(mask)  # int

        return this_number_of_inliers, this_inlier_indices

    def one_iter_ransac(pairs_before, pairs_current):
        """
        Run a single iteration of RANSAC using three random feature pairs from the previous and current frames.

        Parameters:
        p_before    : numpy.ndarray     size: (number_of_pairs, 3)
        p_cur       : numpy.ndarray     size: (number_of_pairs, 3)
        The np arrays are each the unfiltered 3D point clouds of the features from the previous and current frame.

        Returns:
        iteration_transform_matrix           : numpy.ndarray     size: (4, 4)
        number_of_inliers     : int           value <= number_of_pairs

        Returns the transform applied to the previous frame, as well as the number of inliers found
        as a result of the transformation.
        """
        rand_idx = np.random.choice(
            number_of_pairs, 3, replace=False
        )  # 3 random samples
        prev_test_points = pairs_before[rand_idx]
        cur_test_points = pairs_current[rand_idx]
        this_temp_transform_matrix = point_cloud_alignment_matrix(
            prev_test_points, cur_test_points
        )
        this_number_of_inliers, _ = inliers_from_t(
            this_temp_transform_matrix, pairs_before, pairs_current
        )
        return this_temp_transform_matrix, this_number_of_inliers

    print(f"starting ransac - maximum number of iteration number: {max_iter}")
    for i in range(max_iter):
        iteration_transform_matrix, number_of_inliers = one_iter_ransac(p_before, p_cur)
        if number_of_inliers > max_inliers:
            max_inliers = number_of_inliers
            top_t = iteration_transform_matrix
            if (number_of_inliers / number_of_pairs) > high_inlier_perc_threshold:
                print(f"finished ransac with {i} iterations, inliers > 95%")
                break
        if i == (max_iter - 1):
            print(f"finished ransac after reaching the maximum number of iterations")

    _, inlier_indices = inliers_from_t(top_t, p_before, p_cur)  # repeat best transform

    p_a = p_before[inlier_indices]
    p_b = p_cur[inlier_indices]

    return p_a, p_b, inlier_indices


class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)  # current rotation (initiated to be eye matrix)
        self.r = np.zeros((3, 1))  # current translation (initiated to be zeros)
        self.kp_l_prev = None  # previous key points (left)
        self.des_l_prev = None  # previous descriptor for key points (left)
        self.kp_r_prev = None  # previous key points (right)
        self.des_r_prev = None  # previous descriptor key points (right)
        self.detector = cv.SIFT_create()  # using sift for detection
        self.feature_color = (255, 191, 0)
        self.inlier_color = (32, 165, 218)

    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img, kp, None)
        return kp, des, feature_image

    def convert_features_to_3d(self, features_corr):
        """
        Take the u and v pixel information (x and y) of the left and right camera from the previous and current
        frames and produce the 3D point cloud points for the previous frame and current frame, independently.

        Parameters:
        features_corr : list
        The features_corr is of size N x 8 where each row is the detected feature and the columns follow
        the following order:
            [prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y]

        Returns:
        p_before: numpy.ndarray
        p_cur: numpy.ndarray
        Each array is of size N x 3 and represents the 3D coordinates of each detected feature, where each array
        represents the respective timeframe.
        """

        # assuming epi-polar from starter code, only look at x coords
        f_prev = features_corr[:, :4]  # N x 4
        f_cur = features_corr[:, 4:8]  # N x 4

        def inverse_stereo(
            stereo_camera_object, l_and_r_feats
        ):  # input N x 4 (l_u, l_v, r_u, r_l)
            """
            Convert features from left and right camera to 3d points using the inverse stereo model

            Parameters:
            features_corr : list
            The features_corr is of size N x 4 where each row is the detected feature and the columns follow
            the following order:
                [l_u, l_v, r_u, r_v]
            Returns:
            numpy.ndarray
                An array of size N x 3 representing the 3D coordinates of the features
            """
            l_u = l_and_r_feats[:, 0]
            disparity = (
                l_u - l_and_r_feats[:, 2]
            )  # assumes epi-polar from starter code filtering and rectification, l_u - r_u
            z = np.array(
                stereo_camera_object.cam.baseline
                * stereo_camera_object.cam.f_len
                / disparity
            )  # N,
            x = np.array(
                z * (l_u - stereo_camera_object.cam.cu) / stereo_camera_object.cam.f_len
            )  # N,
            y = np.array(
                z
                * (l_and_r_feats[:, 1] - stereo_camera_object.cam.cv)
                / stereo_camera_object.cam.f_len
            )  # N,
            points_3d = np.column_stack((x, y, z))
            return points_3d  # N x 3 (x, y, z)

        p_before = inverse_stereo(self, f_prev)
        p_cur = inverse_stereo(self, f_cur)
        return p_before, p_cur  # each N x 3

    # POSE ESTIMATION ################################################

    def pose_estimation(self, features_corr):
        """
        Process the detected features of the left and right camera in the previous and current frames as 3D
        point clouds, identify the features that are inliers, and use those features to compute the rotation
        and translation from the previous frame to the current frame.

        Parameters:
        features_corr : list
        The features_corr is of size N x 8 where each row is the detected feature and the columns follow
        the following order:
            [prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y]

        Returns:
        rotation_matrix           : numpy.ndarray     size: (3, 3)
        r           : numpy.ndarray     size: (3, 1)
        f_r_prev    : list              size: (M, 2)
        f_r_cur     : list              size: (M, 2)

        Rotation_matrix and r are the translations applied to the previous frame to convert them to the current_frame.
        F_r_prev are the filtered features for the right camera in the previous frame, and f_r_cur are
        the filtered features for the right camera in the current frame.
        """

        # ------------- start your code here -------------- #

        # 1. Convert to 3d cloud points, two sets: before: [x, y, z], and current: [x, y, z],
        # 2. Iteratively run RANSAC to get inliers by fitting rotation_matrix and r to 3 points,
        # 3. Choose a version that had the most inliers,
        # 4. Use M features for cloud alignment
        p_before, p_cur = self.convert_features_to_3d(
            features_corr
        )  # np arrays N x 3 each
        certainty = 0.99
        max_iter = compute_max_iter_ransac(certainty)
        p_a, p_b, inliers_indices = filter_inliers_ransac(
            p_before, p_cur, max_iter
        )  # M x 3 arrays

        transform_matrix = point_cloud_alignment_matrix(p_a, p_b)
        rotation_matrix = transform_matrix[:3, :3]
        r = transform_matrix[:3, 3]

        filtered_features_corr = features_corr[inliers_indices]
        f_r_prev, f_r_cur = (
            filtered_features_corr[:, 2:4],
            filtered_features_corr[:, 6:8],
        )
        return rotation_matrix, r, f_r_prev, f_r_cur

    def process_first_frame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right

    def process_second_frame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        # compute feature correspondence
        features_corr = find_feature_correspondences(
            self.kp_l_prev,
            self.des_l_prev,
            self.kp_r_prev,
            self.des_r_prev,
            kp_l,
            des_l,
            kp_r,
            des_r,
        )  # Nx8
        # draw the feature tracking on the left img
        img_l_tracking = feature_tracking(
            features_corr[:, 0:2],
            features_corr[:, 4:6],
            img_left,
            color=self.feature_color,
        )

        # lab4 assignment: compute the vehicle pose
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_corr)

        # draw the feature (inliers) tracking on the right img
        img_r_tracking = feature_tracking(
            f_r_prev, f_r_cur, img_right, color=self.inlier_color, alpha=1.0
        )

        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME

        return img_l_tracking, img_r_tracking

    def process_frame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)

        # compute feature correspondence
        features_corr = find_feature_correspondences(
            self.kp_l_prev,
            self.des_l_prev,
            self.kp_r_prev,
            self.des_r_prev,
            kp_l,
            des_l,
            kp_r,
            des_r,
        )  # N x 8
        # draw the feature tracking on the left img
        img_l_tracking = feature_tracking(
            features_corr[:, 0:2],
            features_corr[:, 4:6],
            img_left,
            color=self.feature_color,
        )

        # lab4 assignment: compute the vehicle pose
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_corr)

        # draw the feature (inliers) tracking on the right img
        img_r_tracking = feature_tracking(
            f_r_prev, f_r_cur, img_right, color=self.inlier_color, alpha=1.0
        )

        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking

    def update(self, img_left, img_right):

        self.new_frame_left = img_left
        self.new_frame_right = img_right

        if self.frame_stage == STAGE_DEFAULT_FRAME:
            frame_left, frame_right = self.process_frame(img_left, img_right)

        elif self.frame_stage == STAGE_SECOND_FRAME:
            frame_left, frame_right = self.process_second_frame(img_left, img_right)

        else:
            frame_left, frame_right = self.process_first_frame(img_left, img_right)

        self.last_frame_left = self.new_frame_left
        self.last_frame_right = self.new_frame_right

        return frame_left, frame_right
