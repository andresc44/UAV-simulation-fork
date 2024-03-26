"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
import numpy as np
import cv2 as cv
import sys
import math

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

# np.random.rand(1217) In starter code, probably typo
np.random.seed(1217)


class StereoCamera:
    def __init__(self, baseline, focalLength, fx, fy, cu, cv):
        self.baseline = baseline
        self.f_len = focalLength
        self.fx = fx
        self.fy = fy
        self.cu = cu
        self.cv = cv

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)                               # current rotation    (initiated to be eye matrix)
        self.r = np.zeros((3,1))                         # current translation (initiated to be zeros)
        self.kp_l_prev  = None                           # previous key points (left)
        self.des_l_prev = None                           # previous descriptor for key points (left)
        self.kp_r_prev  = None                           # previous key points (right)
        self.des_r_prev = None                           # previoud descriptor key points (right)
        self.detector = cv.xfeatures2d.SIFT_create()     # using sift for detection
        self.feature_color = (255, 191, 0)
        self.inlier_color = (32,165,218)

            
    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img,kp,None)
        return kp, des, feature_image

    def featureTracking(self, prev_kp, cur_kp, img, color=(0,255,0), alpha=0.5):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cover = np.zeros_like(img)
        # Draw the feature tracking 
        for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
            a, b = new.ravel()
            c, d = old.ravel()  
            a,b,c,d = int(a), int(b), int(c), int(d)
            cover = cv.line(cover, (a,b), (c,d), color, 2)
            cover = cv.circle(cover, (a,b), 3, color, -1)
        frame = cv.addWeighted(cover, alpha, img, 0.75, 0)
        
        return frame
    
    def find_feature_correspondences(self, kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r):
        VERTICAL_PX_BUFFER = 1                                # buffer for the epipolor constraint in number of pixels
        FAR_THRESH = 7                                        # 7 pixels is approximately 55m away from the camera 
        CLOSE_THRESH = 65                                     # 65 pixels is approximately 4.2m away from the camera
        
        nfeatures = len(kp_l)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)        # BFMatcher for SIFT or SURF features matching

        ## using the current left image as the anchor image
        match_l_r = bf.match(des_l, des_r)                    # current left to current right
        match_l_l_prev = bf.match(des_l, des_l_prev)          # cur left to prev. left
        match_l_r_prev = bf.match(des_l, des_r_prev)          # cur left to prev. right

        kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
        kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
        kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

        kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
        kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
        kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

        ## loop through all the matched features to find common features
        features_coor = np.zeros((1,8))
        for pt_idx in np.arange(nfeatures):
            if (pt_idx in set(kp_query_idx_l_r)) and (pt_idx in set(kp_query_idx_l_l_prev)) and (pt_idx in set(kp_query_idx_l_r_prev)):
                temp_feature = np.zeros((1,8))
                temp_feature[:, 0:2] = kp_l_prev[kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]].pt 
                temp_feature[:, 2:4] = kp_r_prev[kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]].pt 
                temp_feature[:, 4:6] = kp_l[pt_idx].pt 
                temp_feature[:, 6:8] = kp_r[kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]].pt 
                features_coor = np.vstack((features_coor, temp_feature))
        features_coor = np.delete(features_coor, (0), axis=0)

        ##  additional filter to refine the feature coorespondences
        # 1. drop those features do NOT follow the epipolar constraint
        features_coor = features_coor[
                    (np.absolute(features_coor[:,1] - features_coor[:,3]) < VERTICAL_PX_BUFFER) &
                    (np.absolute(features_coor[:,5] - features_coor[:,7]) < VERTICAL_PX_BUFFER)]

        # 2. drop those features that are either too close or too far from the cameras
        features_coor = features_coor[
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) < CLOSE_THRESH)]

        features_coor = features_coor[
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) < CLOSE_THRESH)]
        # features_coor:
        #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
        return features_coor
    
    ### POINT CLOUD ALIGNMENT ###################################

    def compute_transformation_matrix(self, points_before, points_after):
        """
        ...
        
        Parameters:
        points_before   : numpy.ndarray   size: (?, 3)      either (M x 3) or (3 x 3)
        points_after    : numpy.ndarray   size: (?, 3)      either (M x 3) or (3 x 3)
        ...
        
        Returns:
        T   : numpy.ndarray   size: (4, 4)
        ...
        """
        # TODO:
        # each input array is ? x 3 (X, Y, Z)
        # compute transform T from array
        T = np.eye(4)
        return T



    ## RANSAC ##################################################
    #TODO:  - Determine good INLIER_THRESHOLD_EUCLIDEAN value in filter_inliers_RASAC fxn
    #       - Determine good HIGH_INLIER_PERC_THRESHOLD value in compute_max_iter_RANSAC fxn
    #       - Determine good perc_outlier value in compute_max_iter_RANSAC fxn
    #       - Debug
    
    def convert_features_to_3D(self, features_coor):
        """
        Take the u and v pixel information (x and y) of the left and right camera from the previous and current
        frames and produce the 3D point cloud points for the previous frame and current frame, independently.

        Parameters:
        features_coor : list
        The features_coor is of size N x 8 where each row is the detected feature and the columns follow 
        the following order:
            [prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y]
        
        Returns:
        p_before: numpy.ndarray
        p_cur: numpy.ndarray
        Each array is of size N x 3 and represent the 3D coordinates of each detected feature, where each array 
        represents the respective timeframe.
        """
        
        #assuming epipolar from starter code, only look at x coords
        f_prev = features_coor[:, :4] #N x 4
        f_cur = features_coor[:, 4:8] #N x 4

        def inverse_stereo(self, l_and_r_feats): #input N x 4 (l_u, l_v, r_u, r_l)
            """
            Convert features from left and right camera to 3d points using the inverse stereo model

            Parameters:
            features_coor : list
            The features_coor is of size N x 4 where each row is the detected feature and the columns follow 
            the following order:
                [l_u, l_v, r_u, r_v]
            Returns:
            numpy.ndarray
                An array of size N x 3 representing the 3D coordinates of the features
            """
            l_u = l_and_r_feats[:, 0]
            disparity = l_u - l_and_r_feats[:, 2] #assumes epipolar from starter code filtering and rectification, l_u - r_u
            Z = np.array(self.cam.baseline * self.cam.f_len / disparity) #N, 
            X = np.array(Z * (l_u - self.cam.cu) / self.cam.f_len) #N,
            Y = np.array(Z * (l_and_r_feats[:, 1] - self.cam.cv) / self.cam.f_len) #N,
            points_3D = np.column_stack((X, Y, Z))
            return points_3D # N x 3 (x, y, z)
        
        p_before = inverse_stereo(self, f_prev)
        p_cur = inverse_stereo(self, f_cur)
        return p_before, p_cur #each N x 3
    
    def compute_max_iter_RANSAC(self, certainty):
        """
        Use the RANSAC eqn to determine a max number of iterations to perform to gain a degree of certainty in
        inlier detection. max_iter = frac{\log(1 - p)}{\log(1 - (1-e)^s)}, where p is the certainty, e
        is the estimated proportion of outliers, and s is the number of samples needed to estimate the model.

        Parameters:
        certainty : float
        value from 0 - 1 (non-inclusive) of how certain we want to be. Usually set to 0.99 or higher
        
        Returns:
        int
        The number of iterations required to achieve the desired certainty
        """
        pts_picked = 3
        perc_outlier = 0.3 ## CHOSEN ARBITRARILY, CONSIDER REVISING#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        N = math.log(1 - certainty) / math.log(1 - (1 - perc_outlier) ** pts_picked)
        return math.ceil(N)
    
    def calculate_euclid(self, p_cur, p_transformed):
        """
        Determine the Euclidean distance of each feature between the current frame, and the transformation
        applied to the previous frame.

        Parameters:
        p_cur           : numpy.ndarray   size: (N, 3)
        p_transformed   : numpy.ndarray   size: (N, 3)
        2 supposedly similar frames of features.
        
        Returns:
        numpy.ndarray   size: (N,)
        Each element represents the Euclidean distance for that row between the 2 frames
        """
        squared_diff = (p_cur - p_transformed) ** 2
        sum_squared_diff = np.sum(squared_diff, axis=1)
        dist = np.sqrt(sum_squared_diff)
        return dist
    
    def filter_inliers_RASAC(self, p_before, p_cur, max_iter):
        """
        Apply RANSAC inlier filtering to the given data to only preserve the features that are inliers.

        Parameters:
        p_before    : numpy.ndarray   size: (N, 3)
        p_cur       : numpy.ndarray   size: (N, 3)
        max_iter    : int
        
        The np arrays are each the unfiltered 3D point clouds of the features from the previous and current frame. 
        max_iter is the number of iterations of RANSAC required to achieve a certain confidence level (0.99).
        
        Returns:
        p_a             : numpy.ndarray   size: (M, 3)
        p_b             : numpy.ndarray   size: (M, 3)
        inlier_indices  : numpy.ndarray   size: (M,)
        
        N >= M. The returned arrays are the 3D point clouds of the inlier features for the previous and current
        frames, respectively. The elements of inlier_indices represent the indices of N that were considered inliers.
        """

        max_inliners = -1
        top_T = np.zeros((4,4))
        INLIER_THRESHOLD_EUCLIDEAN = 10 # metres CONSIDER REVISING !!!!!!!!!!!!!!!!!!!!!!!!!
        HIGH_INLIER_PERC_THRESHOLD = 0.95 #more than HIGH_INLIER_PERC_THRESHOLD percent of points are inliers, can break RANSAC
        N = p_before.shape[0]

        def inliers_from_T(T, p_before, p_cur):
            """
            Given a certain transform T, apply that transform to the previous frame pointcloud and determine
            the inliers by comparing the euclidean distance to the current frame pointcloud.

            Parameters:
            T           : numpy.ndarray     size: (4, 4)
            p_before    : numpy.ndarray     size: (N, 3)
            p_cur       : numpy.ndarray     size: (N, 3)
            
            The np arrays are each the unfiltered 3D point clouds of the features from the previous and current frame.

            Returns:
            inliers_cnt     : int           value <= N
            inlier_indices  : numpy.ndarray size: (inliers_cnt,) and each element is value from 0-> N-1
            
            Returns the number of inliers found, as well as the indices of those inliers w.r.t. the original
            N features.
            """
            p_before_T = np.hstack((p_before, np.ones((N, 1)))).T #4 x N where last row is 1s

            p_transformed_T = T @ p_before_T # 4x4 @ 4 x N => 4 x N
            p_transformed = p_transformed_T[:3, :].T #N x 3     X, Y, Z

            dist = self.calculate_euclid(p_cur, p_transformed) #N,
            mask = dist < INLIER_THRESHOLD_EUCLIDEAN #Boolean len N
            inlier_indices  = np.where(mask)[0] # M,
            inliers_cnt = np.count_nonzero(mask) #int

            return inliers_cnt, inlier_indices
            
        def one_iter_RANSAC(p_before, p_cur):
            """
            Run a single iteration of RANSAC using 3 random feature pairs from the previous and current frames.

            Parameters:
            p_before    : numpy.ndarray     size: (N, 3)
            p_cur       : numpy.ndarray     size: (N, 3)
            The np arrays are each the unfiltered 3D point clouds of the features from the previous and current frame.

            Returns:
            temp_T           : numpy.ndarray     size: (4, 4)
            inliers_cnt     : int           value <= N
            
            Returns the transform that was applied to the previous frame, as well as the number of inliers found
            as a result of the transformation.
            """
            rand_idx = np.random.choice(N, 3, replace=False) #3 random samples
            prev_test_points = p_before[rand_idx]
            cur_test_points = p_cur[rand_idx]
            temp_T = self.compute_transformation_matrix(prev_test_points, cur_test_points)
            inliers_cnt, _ = inliers_from_T(temp_T, p_before, p_cur)
            return temp_T, inliers_cnt

        for _ in range(max_iter):
            temp_T, inliers_cnt = one_iter_RANSAC(p_before, p_cur)
            if inliers_cnt > max_inliners:
                max_inliners = inliers_cnt
                top_T = temp_T
                if (inliers_cnt / N) > HIGH_INLIER_PERC_THRESHOLD:
                    break

        _, inlier_indices = inliers_from_T(top_T, p_before, p_cur) #repeat best transform

        p_a = p_before[inlier_indices]
        p_b = p_cur[inlier_indices]

        return p_a, p_b, inlier_indices
    


    ## POSE ESTIMATION ################################################
    
    def pose_estimation(self, features_coor):
        """
        Process the detected features of the left and right camera in the previous and current frames as 3D 
        pointclouds, identify the features that are inliers, and use those features to compute the rotation 
        and translation from the previous frame to the current frame.

        Parameters:
        features_coor : list
        The features_coor is of size N x 8 where each row is the detected feature and the columns follow 
        the following order:
            [prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y]
        
        Returns:
        C           : numpy.ndarray     size: (3, 3)
        r           : numpy.ndarray     size: (3,)
        f_r_prev    : list              size: (M, 2)
        f_r_cur     : list              size: (M, 2)

        C and r are the translations applied to the previous frame to convert them to the current_frame.
        f_r_prev are the filtered features for the right camera in the previous frame, and f_r_cur are 
        the filtered features for the right camera in the current frame.
        """


        # ------------- start your code here -------------- #

        #1. Convert to 3d pointcloud points. 2 sets before: [x, y, z], and current: [x, y, z]
        #2. Iteratively run RANSAC to get inliers by fitting C and r to 3 points
        #3. Choose version that had the most inliers
        #3. Use M features for cloud alignment
        p_before, p_cur = self.convert_features_to_3D(features_coor) #np arrays N x 3 each
        certainty = 0.99
        max_iter = self.compute_max_iter_RANSAC(certainty)
        p_a, p_b, inlier_indices = self.filter_inliers_RASAC(p_before, p_cur, max_iter) #M x 3 arrays
        

        #TODO: Compute C and r from p_a and p_b which are inlier points
        T = self.compute_transformation_matrix(p_a, p_b)         
        C = T[:3, :3]
        r = T[:3, 3]


        filtered_features_coor = features_coor[inlier_indices]
        f_r_prev, f_r_cur = filtered_features_coor[:,2:4], filtered_features_coor[:,6:8]
        return C, r, f_r_prev, f_r_cur
    
    def processFirstFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        
        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right
    
    def processSecondFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
    
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r) #Nx8
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6],img_left, color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right, color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME
        
        return img_l_tracking, img_r_tracking

    def processFrame(self, img_left, img_right, frame_id):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)

        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r) #N x 8
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right,  color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking
    
    def update(self, img_left, img_right, frame_id):
               
        self.new_frame_left = img_left
        self.new_frame_right = img_right
        
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            frame_left, frame_right = self.processFrame(img_left, img_right, frame_id)
            
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            frame_left, frame_right = self.processSecondFrame(img_left, img_right)
            
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            frame_left, frame_right = self.processFirstFrame(img_left, img_right)
            
        self.last_frame_left = self.new_frame_left
        self.last_frame_right= self.new_frame_right
        
        return frame_left, frame_right 


