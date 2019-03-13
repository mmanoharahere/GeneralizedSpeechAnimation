#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import pickle as pkl
from keras.callbacks import TensorBoard
from time import time
from operator import itemgetter as ig
from config import *
#########################################################################################
#define a class to get trimmed videos in the entire folder

class trainModel:

    def __init__(self):
         self.time_delay = time_delay
         #initiliaze audio extraction
         self.look_back = look_back
         self.n_epoch = train_epoch
         self.n_videos = train_n_videos
         
         
    
    def load_pickle_files(self):
        with open('data/audio_kp_full/audio_kp22847.pickle', 'rb') as pkl_file:
            audio_kp = pkl.load(pkl_file)
        with open('data/pca_full/pkp22847.pickle', 'rb') as pkl_file:
            video_kp = pkl.load(pkl_file)
        with open('data/pca_full/pca22847.pickle', 'rb') as pkl_file:
            pca = pkl.load(pkl_file)
        
        return audio_kp,video_kp,pca
    
    def get_common_keys(self,audio_kp,video_kp):
        keys_audio = audio_kp.keys()
        keys_video = video_kp.keys()
        keys = sorted(list(set(keys_audio).intersection(set(keys_video))))
        
        return keys
    
    def scale_data(self,X):
        scalerX = MinMaxScaler(feature_range=(0, 1))
        X = scalerX.fit_transform(X)
        return X
    
    
    def get_kp_range(self,X,y,keys,audio_kp,video_kp):
        for key in tqdm(keys[0:self.n_videos]):
            audio = audio_kp[key]
            video = video_kp[key]
            if (len(audio) > len(video)):
                audio = audio[0:len(video)]
            else:
                video = video[0:len(audio)]
            start = (self.time_delay-self.look_back) if (self.time_delay-self.look_back > 0) else 0
            for i in range(start, len(audio)-self.look_back):
                a = np.array(audio[i:i+self.look_back])
                v = np.array(video[i+self.look_back-self.time_delay]).reshape((1, -1))
                X.append(a)
                y.append(v)
        return X,y
                
     
    def create_model(self):
        model = Sequential()
        model.add(LSTM(60, input_shape=(self.look_back, 26)))
        model.add(Dropout(0.25))
        model.add(Dense(20))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        return model
    
    def run(self):
        audio_kp,video_kp,pca = self.load_pickle_files()
        c = list(audio_kp.keys())
        d = c[:numOfFiles] #19087 -- change in config
    
        audio_kp = dict(zip(d, ig(*d)(audio_kp)))
            
        keys = self.get_common_keys(audio_kp,video_kp)
        X,y = [],[]
        
        X,y = self.get_kp_range(X,y,keys,audio_kp,video_kp)
        X = np.array(X)
        y = np.array(y)
        shapeX = X.shape
        shapey = y.shape
        #print('Shapes:', X.shape, y.shape)
        X = X.reshape(-1, X.shape[2])
        y = y.reshape(-1, y.shape[2])
        #print('Shapes:', X.shape, y.shape)
        
        scalerX = MinMaxScaler(feature_range=(0, 1))
        scalery = MinMaxScaler(feature_range=(0, 1))
            
        scaler_transform = scalery.fit(y)
        
        with open('data/pca_full/scalery22847.pickle', 'wb') as pkl_file:
        	pkl.dump(scaler_transform, pkl_file)
            
        
        X = scalerX.fit_transform(X)
        y = scaler_transform.transform(y)
        
        
        X = X.reshape(shapeX)
        y = y.reshape(shapey[0], shapey[2])
        
        print('Shapes:', X.shape, y.shape)
        print('X mean:', np.mean(X), 'X var:', np.var(X))
        print('y mean:', np.mean(y), 'y var:', np.var(y))
        
        split1 = int(0.8*X.shape[0])
        split2 = int(0.9*X.shape[0])
        
        train_X = X[0:split1]
        train_y = y[0:split1]
        val_X = X[split1:split2]
        val_y = y[split1:split2]
        test_X = X[split2:]
        test_y = y[split2:]
        
        tbCallback = TensorBoard(log_dir="logs/{}".format(time()))
        checkpoint = ModelCheckpoint('my_model_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        model = self.create_model()
        model.fit(train_X, train_y, epochs=self.n_epoch, batch_size=4, verbose=1, shuffle=True, callbacks=[tbCallback, checkpoint], validation_data=(val_X, val_y))
        test_error = np.mean(np.square(test_y - model.predict(test_X)))
        print('Test Error: ', test_error)
        
        model.save('my_model.h5')
        model.save_weights('my_model_weights.h5')
        print('Saved Model.')
        
        
        return        

#read path from the user to the data folder
if __name__== "__main__":
    trainModel_obj = trainModel()
    trainModel_obj.run()
    print("Model is training!") 
    
