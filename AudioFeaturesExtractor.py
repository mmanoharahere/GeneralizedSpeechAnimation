#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:46:23 2018

@author: shruti
"""

from webvtt import WebVTT
import subprocess
import os
import sys

import glob
from tqdm import tqdm
import soundfile as sf
import scipy.io.wavfile as wav
from python_speech_features import logfbank
import pickle as pkl

from config import *

#define a class to get trimmed videos in the entire folder

class AudioFeatureExtractor:
    
    def __init__(self):
         self.number_of_videos = number_of_videos
         #initiliaze audio extraction
         self.trimmed_video_path = trimmed_videos_path
         self.audio_path = audio_path
         self.audio_kp_path = audio_kp_path
         #check if the path exists
         if not(os.path.exists(self.audio_path)):
             subprocess.call('mkdir -p ' + self.audio_path, shell=True)   
         
         return
     
    def convert_to_audio(self):
        trimmed_videos_list = sorted(glob.glob(self.trimmed_video_path+'/*.mp4'))
        for input_file in trimmed_videos_list:
            command = 'ffmpeg -i ' + input_file + ' -ab 160k -ac 1 -ar 16000 -vn ' + self.audio_path + input_file[len(self.trimmed_video_path): -len('.mp4')] + '.wav'
            subprocess.call(command, shell=True)
        print("========================Audios from the trimmed videos have been extracted successfully==================")
        return
    
    def make_kp_folders(self):
        #audios are now in 16k sampling freq and .wav files in the audio path -- make separate keypoints folders for every extracted audio file
        
        audio_file_list = sorted(glob.glob( self.audio_path+'*.wav'))
        if not(os.path.exists(self.audio_kp_path)):
            subprocess.call('mkdir -p ' + self.audio_kp_path, shell=True)
        return audio_file_list
    
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
        
        
    def extract_audio_features(self):
         self.convert_to_audio()
         file_list = self.make_kp_folders()
         
         #create a dictionary with key as the input audio files and values as logfbank values
         
         d = {}
         resumeFrom = 0
         #frame_rate = 5
         
         for index, input_file in enumerate(tqdm(file_list[resumeFrom:])):
             key = input_file[len(self.audio_path):-len('.wav')]
             
             #Read .wav file
             (sample_rate, signal) = wav.read(input_file)
             
             #extracting logfbank features -- audio features
             features = logfbank(signal, sample_rate)
             d[key] = features
             
             #save the extracted features in a pickle file
             current_Filename = self.audio_kp_path + 'audio_kp' + str(index+resumeFrom+1) + '.pickle'
             old_Filename = self.audio_kp_path + 'audio_kp' + str(index+resumeFrom-2) + '.pickle'
             
             self.save_pickle_files(current_Filename,index,resumeFrom,d)
             
             #delete the not required versions of the pickled audio key features extract file
             self.del_old_pickles(old_Filename)
         return
          

#read path from the user to the data folder
if __name__== "__main__":
    AudioFeatureExtractor_obj = AudioFeatureExtractor()
    AudioFeatureExtractor_obj.extract_audio_features()
    print("Audio features (logfbank) extracted successfully!!") 
    
