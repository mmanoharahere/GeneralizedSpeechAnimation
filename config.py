#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:19:09 2018

@author: shruti
"""

import os


#generate trimmed videos required paths :
dataset_path = 'speech_animation/'
video_path = os.path.join(dataset_path, 'data/videos')
trimmed_videos_path = os.path.join(dataset_path, 'data/trimmed_videos')
captions_path = os.path.join(dataset_path, 'data/captions')

number_of_videos = 303 #4


#audio features extraction required var
audio_path = os.path.join(dataset_path, 'data/audios')
audio_kp_path = os.path.join(dataset_path, 'data/audio_kp')


#video features extract required variables
extract_images_path = os.path.join(dataset_path, 'data/images')
images_kp_path = os.path.join(dataset_path, 'data/images_kp_raw')


#pre-download this -- add this to the requirements file
pre_download_shape_predictor = 'shape_predictor_68_face_landmarks.dat' 


#pca and upsampling file
pca_path = os.path.join(dataset_path, 'pca')
numOfFiles = 19087 #139


#run.py required variables
time_delay = 20
look_back = 50


#test file names
filename = '00302-014' 
#Good videos:    '00300-050', '00221-043', 
#videos:     '00295-001', '00301-097', '00296-014'
#Bad videos:     '00300-039', '00302-014'


inputFile = 'data/audios/' + filename + '.wav'
outputFolder = 'testing_output_images_' + filename + '/' 
model_file = 'my_model.h5'


#train.py required variables
train_epoch = 30
train_n_videos = numOfFiles #2


#test file variables
inputTestFolder = 'data/images/' + filename +'/'
inputToA2KeyModel = 'data/test_data_' + filename + '/images/'
outputFolderGroundTruth ='data/GroundTruth_input_' + filename + '/'
outputFolderKp = 'data/test_data_' + filename + '/'
saveFilename = outputFolderKp + 'kp_test.pickle'
