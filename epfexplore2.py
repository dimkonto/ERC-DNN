# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:21:49 2021

@author: dimkonto
"""
#CREATE THE DATASET OF 241 VARIABLES + OUTPUT VARIABLE
import pandas as pd
import numpy as np
import math
from tabulate import tabulate
from epftoolbox.data import read_data
from epftoolbox.models import DNNModel
from epftoolbox.models import hyperparameter_optimizer
from epftoolbox.models import evaluate_dnn_in_test_dataset
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from numpy import array
from numpy import split
import os


def split_daily(trainset,testset):
    train = array(split(trainset, len(trainset)/24))
    test = array(split(testset, len(testset)/24))
    return train,test

def supervised_conversion(trainset, n_input, n_out=24):
	# flatten data
	data = trainset.reshape((trainset.shape[0]*trainset.shape[1], trainset.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

def lagfeatures(data,featurename, nlags):
    for i in range(nlags):
        shiftpos = i+1
        data[str(featurename)+'lag_'+str(shiftpos)] = data[featurename].shift(shiftpos)
    return data

def dayaheadfeatures(data,featurename, nsteps):
    
    for i in range(nsteps):
        shiftpos = i+1
        data[str(featurename)+'dayahead_'+str(shiftpos)] =np.nan
    
    for j in range(data.shape[0]-24):
        for i in range(nsteps):
            shiftpos = i+1
            data[str(featurename)+'dayahead_'+str(shiftpos)].values[j] = data[featurename].values[i+j]
    return data

path = r'D:\Datasets\epfmodels'
df_train, df_test = read_data(path=path, dataset='NP', years_test=2)


df_train['Date'] = pd.to_datetime(df_train.index,format='%d-%m-%Y %H:%M')
df_test['Date'] = pd.to_datetime(df_test.index,format='%d-%m-%Y %H:%M')
df_train['dayofweek_num']=df_train['Date'].dt.dayofweek
df_test['dayofweek_num']=df_test['Date'].dt.dayofweek

df_train=df_train.drop(['Date'],axis=1)
df_test=df_test.drop(['Date'],axis=1)



print(df_train.head())

###CREATE FEATURE SET OF 241 INPUT FEATURES FOR DNN
#TRAIN SET
df_train = lagfeatures(df_train, featurename='Price', nlags=7*24)
df_train = lagfeatures(df_train, featurename='Exogenous 1', nlags=7*24)
df_train = lagfeatures(df_train, featurename='Exogenous 2', nlags=7*24)

df_train = dayaheadfeatures(df_train, featurename='Exogenous 1', nsteps=24)
df_train = dayaheadfeatures(df_train, featurename='Exogenous 2', nsteps=24)

#TEST SET
df_test = lagfeatures(df_test, featurename='Price', nlags=7*24)
df_test = lagfeatures(df_test, featurename='Exogenous 1', nlags=7*24)
df_test = lagfeatures(df_test, featurename='Exogenous 2', nlags=7*24)

df_test = dayaheadfeatures(df_test, featurename='Exogenous 1', nsteps=24)
df_test = dayaheadfeatures(df_test, featurename='Exogenous 2', nsteps=24)

print(df_train.shape)
print(df_train.head())

for i in range(73,145):
    #train
    df_train = df_train.drop(['Pricelag_'+str(i)],axis=1)
    #test
    df_test = df_test.drop(['Pricelag_'+str(i)],axis=1)
for i in range(25,145):
    #train
    df_train = df_train.drop(['Exogenous 1lag_'+str(i)],axis=1)
    df_train = df_train.drop(['Exogenous 2lag_'+str(i)],axis=1)
    #test
    df_test = df_test.drop(['Exogenous 1lag_'+str(i)],axis=1)
    df_test = df_test.drop(['Exogenous 2lag_'+str(i)],axis=1)
    
    
print(df_train.shape)
#Ensure that the dataset has no NaN values and 241 features + the target variable
#train
df_train = df_train.dropna()
df_train=df_train.drop(columns=['Exogenous 1', 'Exogenous 2'])
#test
df_test = df_test.dropna()
df_test=df_test.drop(columns=['Exogenous 1', 'Exogenous 2'])

print('TRAIN SET SHAPE')    
print(df_train.shape)
print('TEST SET SHAPE')
print(df_test.shape)
#print(df_train['Exogenous 1'].values[0:24])
for i in range(24):
    pos=i+1
    print(df_train['Exogenous 1dayahead_'+str(pos)].values[0])

df_train.to_csv(r'D:\Datasets\epfmodels\NP_train_epf.csv',index=False)
df_test.to_csv(r'D:\Datasets\epfmodels\NP_test_epf.csv',index=False)
#print(df_test.head())


"""
trainset = df_train.values
testset = df_test.values

print(trainset.shape)
print(trainset[:5])
print(testset.shape)
trainset_daily, testset_daily=split_daily(trainset, testset)
print(trainset_daily.shape)
#getting the price column of the 3d numpy array for day 0
#print(trainset_daily[0,:,0])

#trainset_daily_X=np.full(shape=(df_train.shape[0]-23,24,3), fill_value=np.nan)
#trainset_X, trainset_Y=supervised_conversion(trainset_daily, n_input=7*24, n_out=24)
#print(trainset_daily.shape)
#print(trainset_daily[:5])


#GET FEATURES AS DESCRIBED BY EPF PAPER (without a rolling window, taken as a sequence)
for i in range (7,trainset_daily.shape[0]):
    #price of day to be predicted
    priced=trainset_daily[i,:,0]
    print(priced)
    #prices of the previous 3 days
    priced_1=trainset_daily[i-1,:,0]
    priced_2=trainset_daily[i-2,:,0]
    priced_3=trainset_daily[i-3,:,0]
    print(priced_1,priced_2,priced_3)
    #price of the previous week
    priced_7=trainset_daily[i-7,:,0]
    print(priced_7)
    #Exogenous variables for that day
    ex1d=trainset_daily[i,:,1]
    ex2d=trainset_daily[i,:,2]
    print(ex1d,ex2d)
    #historical data from one day and one week ago for exogenous variables
    ex1d_1=trainset_daily[i-1,:,1]
    ex1d_7=trainset_daily[i-7,:,1]
    ex2d_1=trainset_daily[i-1,:,2]
    ex2d_7=trainset_daily[i-7,:,2]
    #dummy var
    dvar=trainset_daily[i,:,3]
    break
"""