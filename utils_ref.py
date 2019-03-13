#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:17:10 2018

@author: mansi
"""

import numpy as np
import dlib
from skimage import io
from imutils import face_utils
import cv2
from config import *

predictor_path = pre_download_shape_predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_sec(time):
    hours, mins , seconds = time.split(':')
    return int(hours) * 3600 + int(mins) * 60 + float(seconds)


def all_landmarks(filename):
    
    image = io.imread(filename)
    locate = detector(image, 1);
    landmarks = np.empty([1,1])
    for i, j in enumerate(locate):
        landmarks = predictor(image, j);
        landmarks = face_utils.shape_to_np(landmarks);
    
    return landmarks

def Tilt(landmarks):
    
    eyes_keypoints = np.array(landmarks[36:48])
    x = eyes_keypoints[:, 0]
    y = -1*eyes_keypoints[:, 1]
    m = np.polyfit(x, y, 1)
    tilt = np.degrees(np.arctan(m[0]))
    
    return tilt

def getKP(keypoints):

    mean_lip_keypoints = np.average(keypoints[48:68], 0)
    keypoints_new = keypoints - mean_lip_keypoints
    x_new = keypoints_new[:, 0]
    y_new = keypoints_new[:, 1]
    theta = np.deg2rad(Tilt(keypoints_new))
    c = np.cos(theta)
    s = np.sin(theta)
    x = x_new*c - y_new*s
    y = x_new*s + y_new*c 
    keypoints_tilt = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
    N = np.linalg.norm(keypoints_tilt, 2)
    
    return [keypoints_tilt/N, N, theta, mean_lip_keypoints]


def drawLips(keypoints, new_img, c = (255, 255, 255), th = 1, show = False):
    
    keypoints = np.float32(keypoints)
    
    for i in range(48, 59):
        cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
    for i in range(60, 67):
        cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
    
    if (show == True):
        cv2.imshow('lol', new_img)
        cv2.waitKey(10000)


