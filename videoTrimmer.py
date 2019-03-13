#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:02:02 2018

@author: shruti
"""

from webvtt import WebVTT
import subprocess
import os
import sys
from utils_ref import *
from config import *

#define a class to get trimmed videos in the entire folder

class videoTrimmer:
    
     def __init__(self):
         self.number_of_videos = number_of_videos
         #initiliaze video width 
         self.video_width = 456
         self.data_path = dataset_path
         #video path in the data directory
         self.video_path = video_path
         self.trimmed_video_path = trimmed_videos_path
         self.captions_path = captions_path
         
         
         #check if the path exists
         if not(os.path.exists(self.trimmed_video_path)):
             subprocess.call('mkdir -p ' + self.trimmed_video_path, shell=True)   
         return
     
     def get_caption_timings(self,caption):
         start = caption.start
         end = caption.end
         t = int(round(get_sec(end) - get_sec(start)))
         return start,end,t
    
     def save_trimmed_videos(self,input_filename,start,end,t,output_filename):
         cmd = 'ffmpeg -i ' + input_filename + '-vf scale='+ str(self.video_width) + ':256 ' + '-ss ' + str(start) + ' -t ' + str(t) + ' -acodec copy ' + output_filename
         subprocess.call(cmd, shell=True)
         return 
    
     def generate_trimmed_videos(self):
         
         for i in range(2, self.number_of_videos):
            print("Video number--------------", i)
            num = str(i).rjust(5, '0')
            captions_filename = self.captions_path+ num +'.en.vtt'
            input_filename = self.video_path + num + '.mp4 '
            captions = WebVTT().read(captions_filename)
            
            print('Total Length of captions is', len(captions))
            
            for index, caption in enumerate(captions):
                
                output_filename = self.trimmed_video_path + num + '-' + str(index).rjust(3, '0') + '.mp4'
                print(output_filename)
                start,end,t = self.get_caption_timings(caption)
                self.save_trimmed_videos(input_filename,start,end,t,output_filename)
         return
          

#read path from the user to the data folder
if __name__== "__main__":
    videoTrimmer_obj = videoTrimmer()
    videoTrimmer_obj.generate_trimmed_videos()
    print("All the videos have been trimmed successfully!")
    
