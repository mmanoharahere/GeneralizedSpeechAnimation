#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:03:29 2018

@author: shruti
"""

from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pickle as pkl

import cv2
import scipy.io.wavfile as wav
from python_speech_features import logfbank
import subprocess
import imageio
import os
import glob

from config import *
#########################################################################################
#define a class to get trimmed videos in the entire folder

class runModel:
    
    def __init__(self):
         self.time_delay = time_delay
         #initiliaze audio extraction
         self.look_back = look_back
         self.outputFolder = outputFolder
         self.model = load_model(model_file)
         self.inputFile = inputFile
         #check if the path exists
         #if it exists -- delete the previously saved one and save a new one.
         
         if(os.path.exists(self.outputFolder)):
             cmd = 'rm -rf '+self.outputFolder + '&& mkdir ' + self.outputFolder
             subprocess.call(cmd ,shell=True)
         if not(os.path.exists(self.outputFolder)):
             subprocess.call('mkdir -p ' + self.outputFolder, shell=True)  
    
    def get_audio_features(self,X):
        (rate, sig) = wav.read(self.inputFile)
        audio = logfbank(sig,rate)
        
        start = ( self.time_delay- self.look_back) if ( self.time_delay- self.look_back > 0) else 0
        for i in range(start, len(audio)- self.look_back):
             a = np.array(audio[i:i+ self.look_back])
             X.append(a)
        
        X = np.array(X)
        shapeX = X.shape
        X = X.reshape(-1, X.shape[2])
        print('Shapes:', X.shape)
        
        return X,shapeX
    
    def scale_data(self,X):
        scalerX = MinMaxScaler(feature_range=(0, 1))
        X = scalerX.fit_transform(X)
        return X
    
    def predict_y(self,X):
        with open('data/pca/scalery22847.pickle', 'rb') as pkl_file: #####change here
            scalery = pkl.load(pkl_file) 
            
        with open('data/pca/pca22847.pickle', 'rb') as pkl_file:
            pca = pkl.load(pkl_file)
            
        y_pred = self.model.predict(X)
        y_pred = scalery.inverse_transform(y_pred)
        y_pred = pca.inverse_transform(y_pred)
        
        return y_pred
   
     
    def subsample(self, y, fps_from = 100.0, fps_to = 29.97):
        factor = int(np.ceil(fps_from/fps_to))
        #subsample the points
        new_y = np.zeros((int(y.shape[0]/factor), 20, 2)) #(timesteps, 20) = (500, 20x2)
        for idx in range(new_y.shape[0]):
            if not (idx*factor > y.shape[0]-1):
			# Get into (x, y) format
                new_y[idx, :, 0] = y[idx*factor, 0:20]
                new_y[idx, :, 1] = y[idx*factor, 20:]
            else:
                break
        new_y = [np.array(each) for each in new_y.tolist()]
        return new_y

    def get_kp_data(self):
        with open('data/test_data_' + filename + '/kp_test.pickle', 'rb') as pkl_file: ######change hereeee
            kp= pkl.load(pkl_file)
        return kp  
    
    def set_kp_length(self, kp, y_pred):
        if (len(kp) < len(y_pred)):
            n = len(kp)
            y_pred = y_pred[:n]
        else:
            n = len(y_pred)
            kp = kp[:n]
            
        return n,kp,y_pred
    
    def getOriginalKeypoints(self,kp_features_mouth, N, tilt, mean):
        # Denormalize the points
        kp_dn = N * kp_features_mouth
        # Add the tilt
        x, y = kp_dn[:, 0], kp_dn[:, 1]
        c, s = np.cos(tilt), np.sin(tilt)
        x_dash, y_dash = x*c + y*s, -x*s + y*c
        kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1,1))))
        # Shift to the mean
        kp = kp_tilt + mean
        return kp
    
    def drawLips(self,keypoints, new_img, c = (255, 255, 255), th = 1, show = False):
        keypoints = np.float32(keypoints)
        for i in range(48, 59):
            cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
        cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
        cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
        cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
        cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
        
        for i in range(60, 67):
            cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
            
        return
        
    def get_mouth_boundary_images(self,y_pred,kp):
        for idx, (x, k) in enumerate(zip(y_pred, kp)):
            
            unit_mouth_kp, N, tilt, mean, unit_kp, keypoints = k[0], k[1], k[2], k[3], k[4], k[5]
            kps = self.getOriginalKeypoints(x, N, tilt, mean)
            keypoints[48:68] = kps
            
            imgfile = 'data/test_data_' + filename + '/images/' + str(idx+1).rjust(5, '0') + '.png'  ####change is required
            im = cv2.imread(imgfile)
            self.drawLips(keypoints, im, c = (255, 255, 255), th = 1, show = False)
            
            
            im_out = np.zeros_like(im)
            im1 = np.hstack((im, im_out))
            im1 = im1[:, :256]
            
            cv2.imwrite(self.outputFolder + str(idx) + '.png', im1)
            
            
        print('Done writing all images')
        
        return
                
    def run(self):
        X = []
        X ,shapeX= self.get_audio_features(X)
        X = self.scale_data(X)
        X = X.reshape(shapeX)
        
        #predict y - pred
        y_pred = self.predict_y(X)
        print('Upsampled number:', len(y_pred))
        y_pred = self.subsample(y_pred, 100, 34)
        print('Subsampled :', len(y_pred))
        
        ##visualization---------------------------------
        kp = self.get_kp_data()
        n,kp,y_pred = self.set_kp_length(kp, y_pred)
        
        ### get the mouth boundaries on images
        self.get_mouth_boundary_images(y_pred, kp)
        return        



#read path from the user to the data folder
if __name__== "__main__":
    runModel_obj = runModel()
    runModel_obj.run()
    print("Images with mouth boundaries on Obama saved!!!!") 
    
    
###Generate video
os.chdir(outputFolder)
with imageio.get_writer('Results.mp4', mode='I', fps=30) as writer:
        for file in sorted(glob.glob("*.png"), key=os.path.getmtime):
            image = imageio.imread(file)
            writer.append_data(image)
print("Video without audio generated!!!!")

os.chdir('..')

input_path = outputFolder + "Results.mp4"
result_path = outputFolder + "Output.mp4"

cmd = "ffmpeg -i '"+input_path+ "' -i '"+ inputFile + "' \
-c:v copy -c:a aac -strict experimental '"+result_path+"'"
subprocess.call(cmd ,shell=True)
print("Video with audio generated!!!")
