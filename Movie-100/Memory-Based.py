#!/usr/bin/env python  
# encoding: utf-8  
""" 
@author: GrH 
@contact: 1271013391@qq.com 
@file: Memory-Based.py 
@time: 2019/2/26 0026 15:22 
"""
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

#读入数据
header=['user_id','item_id','rating','timestamp']
df=pd.read_csv('D:/PycharmProjects/Movie-100/data/u.data',sep='\t',names=header)
n_users=df.user_id.unique().shape[0]
n_items=df.item_id.unique().shape[0]
# print('Numbers of users='+str(n_users),'and Numbers of items='+str(n_items))
#分割数据集
train_data,test_data=cv.train_test_split(df,test_size=0.25)
# create 2 user-item matrices
train_data_matrix=np.zeros((n_users,n_items))
for line in train_data.itertuples():
    #数据中用户和物品是从1开始计算的
    train_data_matrix[line[1]-1,line[2]-1]=line[3]

test_data_matrix=np.zeros((n_users,n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1,line[2]-1]=line[3]

user_similarity=pairwise_distances(train_data_matrix,metric='cosine')
item_similarity=pairwise_distances(train_data_matrix.T,metric='cosine')
#预测函数
def predict(ratings,similarity,type='user'):
    if type=='user':
        mean_user_rating=ratings.mean(axis=1)
        ratings_diff=(ratings-mean_user_rating[:,np.newaxis])
        # print(ratings_diff)
        pred=mean_user_rating[:,np.newaxis]+similarity.dot(ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
    elif type=='item':
        pred=ratings.dot(similarity)/np.array([np.abs(similarity).sum(axis=1)])
    return pred
#输出结果
item_prediction=predict(train_data_matrix,item_similarity,type='item')
np.savetxt('D:/PycharmProjects/Movie-100/item_prediction.csv',item_prediction,delimiter=',')
user_prediction=predict(train_data_matrix,user_similarity,type='user')
np.savetxt('D:/PycharmProjects/Movie-100/user_prediction.csv',user_prediction,delimiter=',')
def rmse(prediction,ground_truth):
    prediction=prediction[ground_truth.nonzero()].flatten()
    ground_truth=ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,ground_truth))

print('user-based CF RMSE='+str(rmse(user_prediction,test_data_matrix)))
print('item-based CF RMSE='+str(rmse(item_prediction,test_data_matrix)))




