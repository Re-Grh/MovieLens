#!/usr/bin/env python  
# encoding: utf-8  
""" 
@author: GrH 
@contact: 1271013391@qq.com 
@file: Data Preparation.py
@time: 2019/3/21 0021 22:14 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns


# Reading ratings file
# Ignore the timestamp column
ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

# Reading users file
users = pd.read_csv('users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])


# # Create a wordcloud of the movie titles
# movies['title'] = movies['title'].fillna("").astype('str')
# title_corpus = ' '.join(movies['title'])
# title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)
#
# # Plot the wordcloud
# plt.figure(figsize=(16,8))
# plt.imshow(title_wordcloud)
# plt.axis('off')
# plt.show()

# Get summary statistics of rating
ratings['rating'].describe()
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
# Display distribution of rating
sns.distplot(ratings['rating'].fillna(ratings['rating'].median()))
# plt.show()

# Join all 3 files into one dataframe
dataset = pd.merge(pd.merge(movies, ratings),users)
# Display 15 movies with highest ratings
print(dataset[['title','genres','rating']].sort_values('rating', ascending=False).head(15))





