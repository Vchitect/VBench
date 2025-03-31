#############################################################################
# code for paper: Sora Generateds Video with Stunning Geometical Consistency
# arxiv: https://arxiv.org/abs/
# Author: <NAME> xuanyili
# email: xuanyili.edu@gmail.com
# github: https://github.com/meteorshowers/SoraGeoEvaluate
#############################################################################
import cv2
import csv
import os
import numpy as np

def EvaluateErrBetweenTwoImage(left_img, right_img, ransac_th):
    left_img = left_img.permute(1, 2, 0) 
    left_img = left_img.numpy().astype(np.uint8)
    img1 = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
    
    right_img = right_img.permute(1, 2, 0) 
    right_img = right_img.numpy().astype(np.uint8)
    img2 = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)

    #  align the size the all picture
    img1 = cv2.resize(img1, (854, 480))
    img2 = cv2.resize(img2, (854, 480))
    
    # 创建SIFT特征点检测器
    num_corners = 40000
    sift = cv2.SIFT_create(num_corners)

    # 在两张图片上检测SIFT特征点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    k=2
    if des1 is None or des2 is None:
        return 0
    if len(des1)<k or len(des2)<k:
        return 0
    # 使用FLANN算法匹配特征点
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=k)

    # 通过距离比率去除错误匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    # 提取正确匹配点的位置
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 匹配点对数
    num_pts = len(src_pts)
    
    # 使用基础矩阵RANSAC去除误匹配
    F, inliers_F = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, ransac_th)
    num_inliers_F = np.sum(inliers_F)
    
    return num_inliers_F