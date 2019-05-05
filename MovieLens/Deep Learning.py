#!/usr/bin/env python  
# encoding: utf-8  
""" 
@author: GrH 
@contact: 1271013391@qq.com 
@file: Deep Learning.py 
@time: 2019/3/25 0025 11:42 
"""
# Import libraries

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Keras libraries
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
# Import CF Model Architecture
from CFModel import CFModel
import warnings
warnings.filterwarnings("ignore")
# Reading ratings file
ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1',
                      usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
max_userid = ratings['user_id'].drop_duplicates().max()
max_movieid = ratings['movie_id'].drop_duplicates().max()

# Reading ratings file
users = pd.read_csv('users.csv', sep='\t', encoding='latin-1',
                    usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading ratings file
movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1',
                     usecols=['movie_id', 'title', 'genres'])
# Create training set
shuffled_ratings = ratings.sample(frac=1.,random_state=0)

# Shuffling users
Users = shuffled_ratings['user_emb_id'].values
# print('Users:', Users, ', shape =', Users.shape)

# Shuffling movies
Movies = shuffled_ratings['movie_emb_id'].values
# print ('Movies:', Movies, ', shape =', Movies.shape)

# Shuffling ratings
Ratings = shuffled_ratings['rating'].values
# print ('Ratings:', Ratings, ', shape =', Ratings.shape)
# Define constants
K_FACTORS = 100 # The number of dimensional embeddings for movies and users
TEST_USER = 2000 # A random test user (user_id = 2000)
# Define model
model = CFModel(max_userid, max_movieid, K_FACTORS)
# Compile the model using MSE as the loss function and the AdaMax learning algorithm
model.compile(loss='mse', optimizer='adamax')

# Callbacks monitor the validation loss
# Save the model weights each time the validation loss has improved
callbacks = [EarlyStopping('val_loss', patience=2),
             ModelCheckpoint('weights.h5', save_best_only=True)]

# Use 20 epochs, 90% training data, 10% validation data
history = model.fit([Users, Movies], Ratings, nb_epoch=20, validation_split=.1, verbose=2, callbacks=callbacks)
# Show the best validation RMSE
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))

