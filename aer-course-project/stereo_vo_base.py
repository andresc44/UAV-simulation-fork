"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
import numpy as np
import cv2 as cv
import sys

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
    
    ## RANSAC ##################################################
    # TODO: - get max iter formula
    #       - add inverse stereo model
    #       - compute T from 3 points
    #       - Apply T and check inliers

    # Order of increasing complexity
    
    def convert_features_to_3D(features_coor):

        f_prev = features_coor[:, :4] #N x 4
        f_cur = features_coor[:, 4:8] #N x 4

        def inverse_stereo(l_and_r_feats):
            ...
            #TODO: calculate 3d coordinates based on left and right features using inverse stereo model
            points_3D = [1, 1, 1]
            return points_3D # N x 3 (x, y, z)
        
        p_before = inverse_stereo(f_prev)
        p_cur = inverse_stereo(f_cur)
        return p_before, p_cur #each N x 3
    
    def compute_max_iter_RANSAC(certainty):
        # Add max iter based on certainty with log fxn
        # TODO: apply the formula to get max iterations
        max_iter = 1000
        return max_iter
    
    def filter_inliers_RASAC(self, p_before, p_cur, max_iter):

        max_inliners = -1
        top_T = np.zeros((4,4))
        INLIER_THRESHOLD_EUCLIDEAN = 10 # metres

        def inliers_from_T(T, p_before, p_cur):
            #Apply C and r to p_before and see which points in p_cur are within tolerance
            # TODO: Apply transformation and check inlier points
            return inliers_cnt, inlier_indices
            
        def one_iter_RANSAC(p_before, p_cur):
            rand_idx = np.random.sample(range(len(p_before)), 3) #3 random samples
            prev_test_points = p_before[rand_idx]
            cur_test_points = p_cur[rand_idx]
            #TODO: Compute T from 3 test points
            T = np.zeros((4, 4))
            inliers_cnt, _ = inliers_from_T(T, p_before, p_cur)
            return T, inliers_cnt

        for _ in range(max_iter):
            temp_T, inliers_cnt = one_iter_RANSAC(p_before, p_cur)
            if inliers_cnt > max_inliners:
                max_inliners = inliers_cnt
                top_T = temp_T

        _, inlier_indices = inliers_from_T(top_T, p_before, p_cur)

        p_a = p_before[inlier_indices]
        p_b = p_cur[inlier_indices]

        return p_a, p_b, inlier_indices
    


    ## POSE ESTIMATION ################################################
    
    def pose_estimation(self, features_coor):
        # # dummy C and r
        # C = np.eye(3)
        # r = np.array([0,0,0])
        # # feature in right img (without filtering)
        # f_r_prev, f_r_cur = features_coor[:,2:4], features_coor[:,6:8]
        # ------------- start your code here -------------- #

        #1. Convert to 3d pointcloud points. 2 sets before: [x, y, z], and current: [x, y, z]
        #2. Iteratively run RANSAC to get inliers by fitting C and r to 3 points
        #3. Choose version that had the most inliers
        #3. Use N' features for cloud alignment
        p_before, p_cur = self.convert_features_to_3D(features_coor)
        certainty = 0.99
        max_iter = self.compute_max_iter_RANSAC(certainty)
        p_a, p_b, inlier_indices = self.filter_inliers_RASAC(p_before, p_cur, max_iter)
        
        
        
        #TODO: Compute C and r from p_a and p_b which are inlier points
        # dummy C and r
        C = np.eye(3)
        r = np.array([0,0,0])



        filtered_features_coor = features_coor[inlier_indices]
        f_r_prev, f_r_cur = filtered_features_coor[:,2:4], filtered_features_coor[:,6:8]
        return C, r, f_r_prev, f_r_cur #for 3rd and 4th argument, return filtered features of size N'
    
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
            #input is prev_l_x, prev_l_y    cur_l_x, cur_l_y,
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        # returns prev_r_x, prev_r_y    cur_r_x, cur_r_y
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # input is prev_r_x, prev_r_y    cur_r_x, cur_r_y
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


