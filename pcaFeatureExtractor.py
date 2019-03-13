#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:39:27 2018

@author: mansi
"""
from webvtt import WebVTT
import subprocess
import os
import numpy as np
import sys

import glob
import tqdm
import cv2

from utils_ref import *
from config import *
from operator import itemgetter as ig
import pickle as pkl
from sklearn.decomposition import PCA


class pcaExtractor:
    
     def __init__(self):
         self.images_kp_path = images_kp_path
         self.pca_path = pca_path
         self.numOfFiles = numOfFiles
         
         #check if the path exists
         if not(os.path.exists(self.pca_path)):
             subprocess.call('mkdir -p ' + self.pca_path, shell=True)
         
         return
     
    
    
     def save_pickle_files(self,saveFilename,index,resumeFrom,d):
        
        if not (os.path.exists(saveFilename)):
            with open(saveFilename, "wb") as output_file:
                pkl.dump(d, output_file)
        
        else:
            with open(saveFilename, "rb") as output_file:
                d = pkl.load(output_file)
                print('Loaded output for ', (index+resumeFrom+1), ' file.')
        return
        
    
     def del_old_pickles(self,oldSaveFilename):
            
        if (os.path.exists(oldSaveFilename)):
            command = 'rm -rf ' + oldSaveFilename
            subprocess.call(command, shell=True)
            
        return
        
        
     def load_custom_pickles(self):
        
        with open(images_kp_path+'kp6013.pickle', 'rb') as kp_file:
            kp1 = pkl.load(kp_file)
    
        with open(images_kp_path+'kp6015.pickle', 'rb') as kp_file:
            kp1_5 = pkl.load(kp_file)
        
        with open(images_kp_path+'kp12251.pickle', 'rb') as kp_file:
            kp2 = pkl.load(kp_file)
            
        with open(images_kp_path+'kp17000.pickle', 'rb') as kp_file:
            kp3 = pkl.load(kp_file)
        
        with open(images_kp_path+'kp22848.pickle', 'rb') as kp_file:
            kp4 = pkl.load(kp_file)
            
        
        video_kp = dict(kp1)
        video_kp.update(kp1_5)
        video_kp.update(kp2)
        video_kp.update(kp3)
        video_kp.update(kp4)
        
        return video_kp
        
     def get_mouth_features(self,big_list):
        new_list=[]
        for key in tqdm.tqdm(sorted(big_list.keys())):
            for frame_kp in big_list[key]:
                kp_mouth = frame_kp[0]
                x = kp_mouth[:, 0].reshape((1, -1))
                y = kp_mouth[:, 1].reshape((1, -1))
                X = np.hstack((x, y)).reshape((-1)).tolist()
                new_list.append(X)
        X = np.array(new_list)
        
        return X
    
     def get_upsampled_features(self,big_list):
        #Upsampling
        print('')
        print('Upsampling...')
        
        # Upsample the lip keypoints
        upsampled_kp = {}
        for key in tqdm.tqdm(sorted(big_list.keys())):
            # print('Key:', key)
            nFrames = len(big_list[key])
            factor = int(np.ceil(100/29.97))
            # Create the matrix
            new_unit_kp = np.zeros((int(factor*nFrames), big_list[key][0][0].shape[0], big_list[key][0][0].shape[1]))
            new_kp = np.zeros((int(factor*nFrames), big_list[key][0][-1].shape[0], big_list[key][0][-1].shape[1]))
        
            # print('Shape of new_unit_kp:', new_unit_kp.shape, 'new_kp:', new_kp.shape)
        
            for idx, frame in enumerate(big_list[key]):
                # Create two lists, one with original keypoints, other with unit keypoints
                new_kp[(idx*(factor)), :, :] = frame[-1]
                new_unit_kp[(idx*(factor)), :, :] = frame[0]
        
                if (idx > 0):
                    start = (idx-1)*factor + 1
                    end = idx*factor
                    for j in range(start, end):
                        new_kp[j, :, :] = new_kp[start-1, :, :] + ((new_kp[end, :, :] - new_kp[start-1, :, :])*(np.float(j+1-start)/np.float(factor)))
                        l = getKP(new_kp[j, :, :])
                        new_unit_kp[j, :, :] = l[0][48:68, :]
        
            upsampled_kp[key] = new_unit_kp
        return upsampled_kp
        
     def extract_pca_features(self):
         #load custom pickle files -- had to merge them coz one computer couldnt handle all the processing
        video_kp = self.load_custom_pickles()
         
        print('Unwrapping all items from the big list')

        c = list(video_kp.keys())
        d = c[:self.numOfFiles]
         
        big_list = dict(zip(d, ig(*d)(video_kp)))
         
         #get mouth features
        X = self.get_mouth_features(big_list)
         
        pca = PCA(n_components=20)
        pca.fit(X)
        with open(self.pca_path + 'pca' + str(self.numOfFiles) + '.pickle', 'wb') as file:
            pkl.dump(pca, file)
            
        with open(self.pca_path + 'explanation' + str(self.numOfFiles) + '.pickle', 'wb') as file:
            pkl.dump(pca.explained_variance_ratio_, file)
            
        print('Explanation for each dimension:', pca.explained_variance_ratio_)
        print('Total variance explained:', 100*sum(pca.explained_variance_ratio_))
         
         
         #get upsampled features -----------
         
        upsampled_kp = {}
        upsampled_kp = self.get_upsampled_features(big_list)
         
        d = {}
         
        keys = sorted(upsampled_kp.keys())
        for key in tqdm.tqdm(keys):
            x = upsampled_kp[key][:, :, 0]
            y = upsampled_kp[key][:, :, 1]
            X = np.hstack((x, y))
            X_trans = pca.transform(X)
            d[key] = X_trans
            
        with open(self.pca_path  + 'pkp' + str(self.numOfFiles) + '.pickle', 'wb') as file:
            pkl.dump(d, file)
            print('Saved Everything')
        return
          

#read path from the user to the data folder
if __name__== "__main__":
    pcaExtractor_obj = pcaExtractor()
    pcaExtractor_obj.extract_pca_features()
    print("PCA features extracted successfully!!") 
    
