#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:34:31 2018

@author: mansi
"""

from webvtt import WebVTT
import subprocess
import os
import numpy as np

import glob
import tqdm
import cv2

from utils_ref import *
import pickle as pkl
from config import *

#define a class to get trimmed videos in the entire folder

class VideoFeatureExtractor:
    def __init__(self):
         self.number_of_videos = number_of_videos
         #initiliaze audio extraction
         self.trimmed_videos_path = trimmed_videos_path
         self.extracted_images_path = extract_images_path
         self.images_kp_path = images_kp_path
        
         #check if the path exists
         if not(os.path.exists(self.extracted_images_path)):
             subprocess.call('mkdir -p ' + self.extracted_images_path, shell=True)
                             
         if not(os.path.exists(self.images_kp_path)):
             subprocess.call('mkdir -p ' + self.images_kp_path, shell=True) 
         
         return

    def save_bmp_files(self,filename, outputpath,num):
        
        command = 'ffmpeg -i ' + filename + ' -vf fps=30 ' + ' -vf scale=-1:256 '+ outputpath + num + '/$filename%05d' + '.bmp'
        subprocess.call(command, shell=True)
        
        return
    
    def crop_images(self,num):
        #read all the bmp images in every num folder of the extracted videos
        imglist = sorted(glob.glob( self.extracted_images_path + num + '/*.bmp'))
        for i in range(len(imglist)):
            img = cv2.imread(imglist[i])
            x = int(np.floor((img.shape[1]-256)/2))
            #Crop image
            crop_img = img[0:256, x:x+256]
            #generate .jpeg files
            cv2.imwrite(imglist[i][0:-len('.bmp')] + '.jpeg', crop_img)
        #Remove .bmp files
        subprocess.call('rm -rf '+ self.extracted_images_path + num + '/*.bmp', shell=True)
        return
    
    def save_landmarks(self,img_directories,resumeFrom):
        d = {}
        for idx, directory in tqdm.tqdm(enumerate(img_directories[resumeFrom:])):
            key = directory[len(self.extracted_images_path):-1]
            imglist = sorted(glob.glob(directory+'*.jpeg'))
            big_list = []
            for file in tqdm.tqdm(imglist):
                keypoints = all_landmarks(file)
                if not (keypoints.shape[0] == 1): 
                    l = getKP(keypoints)
                    unit_kp, N, tilt, mean = l[0], l[1], l[2], l[3]
                    kp_mouth = unit_kp[48:68]
                    store_list = [kp_mouth, N, tilt, mean, unit_kp, keypoints]
                    prev_store_list = store_list
                    big_list.append(store_list)
                else:
                    big_list.append(prev_store_list)
                
                
            d[key] = big_list
        return d
    
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
                
    def extract_video_features(self):
         trimmed_video_list = sorted(glob.glob(self.trimmed_videos_path+'/*.mp4'))
         for index, filename in tqdm.tqdm(enumerate(trimmed_video_list)):
             num = filename[len(self.trimmed_videos_path):-len('.mp4')]
             
             #create a folder to save all the extracted images for every trimmed video inside the extracted images path
             if not(os.path.exists(self.extracted_images_path+num)):
                  subprocess.call('mkdir -p ' + self.extracted_images_path+num, shell=True)
            
             self.save_bmp_files(filename,self.extracted_images_path,num)   
             self.crop_images(num)
             
             resumeFrom = 0
             #get all the image directories in a sorted manner -- it should be stored like this anyway
             img_directories = sorted(glob.glob(self.extracted_images_path+'*/'))
             
             #create a dictionary to store the features
             d = {}
             #save all facial landmarks in a dictonary
             
             d = self.save_landmarks(img_directories, resumeFrom)
             
             #save the extracted features in a pickle file
             
             current_Filename = self.images_kp_path + 'kp' + str(index+resumeFrom+1) + '.pickle'
             old_Filename = self.images_kp_path + 'kp' + str(index+resumeFrom-2) + '.pickle'
             
             self.save_pickle_files(current_Filename,index,resumeFrom,d)
             
             #delete the not required versions of the pickled audio key features extract file
             self.del_old_pickles(old_Filename)
         return

#read path from the user to the data folder
if __name__== "__main__":
    VideoFeatureExtractor_obj = VideoFeatureExtractor()
    VideoFeatureExtractor_obj.extract_video_features()
    print("All the video features have been extracted successfully!")
    
