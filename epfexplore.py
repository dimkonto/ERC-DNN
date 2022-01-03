# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:53:58 2021

@author: dimkonto
"""
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

path = r'D:\Datasets\epfmodels'
df_train, df_test = read_data(path=path, dataset='NP', years_test=2)

print(df_train.head())
#print(df_test.head())


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


###FEATURE ENGINEERING (SAME VARIABLE CREATION WILL OCCUR TO THE TRAINING SET)
#Date features
df_train['Date'] = pd.to_datetime(df_train.index,format='%d-%m-%Y %H:%M')
df_train['year'] = df_train['Date'].dt.year
df_train['month'] = df_train['Date'].dt.month
df_train['day'] = df_train['Date'].dt.day
df_train['dayofweek_num']=df_train['Date'].dt.dayofweek
df_train['dayofweek_name']=df_train['Date'].dt.day_name()

df_train['Hour'] = df_train['Date'].dt.hour
df_train['Quarter'] = df_train['Date'].dt.quarter

#Create Lag Features for the previous day
df_train = lagfeatures(df_train, featurename='Price', nlags=24)
df_train = lagfeatures(df_train, featurename='Exogenous 1', nlags=24)
df_train = lagfeatures(df_train, featurename='Exogenous 2', nlags=24)

fig_acf_price_train=plot_acf(df_train['Price'], lags=168, title= 'Autocorrelation Price - Train')
fig_pacf_price_train=plot_pacf(df_train['Price'], lags=168, title ='Partial Autocorrelation Price - Train' )

fig_acf_price_train.savefig(r'D:\Datasets\epfmodels\charts\fig_acf_price_train.jpg',dpi=300,bbox_inches="tight")
fig_pacf_price_train.savefig(r'D:\Datasets\epfmodels\charts\fig_pacf_price_train.jpg',dpi=300,bbox_inches="tight")

#check autocorrelation for exogenous variables
fig_acf_ex1_train=plot_acf(df_train['Exogenous 1'], lags=168, title= 'Autocorrelation EX1 - Train')
fig_pacf_ex1_train=plot_pacf(df_train['Exogenous 1'], lags=168, title= 'Partial Autocorrelation EX1 - Train')

fig_acf_ex1_train.savefig(r'D:\Datasets\epfmodels\charts\fig_acf_ex1_train.jpg',dpi=300,bbox_inches="tight")
fig_pacf_ex1_train.savefig(r'D:\Datasets\epfmodels\charts\fig_pacf_ex1_train.jpg',dpi=300,bbox_inches="tight")

fig_acf_ex2_train=plot_acf(df_train['Exogenous 2'], lags=168, title= 'Autocorrelation EX2 - Train')
fig_pacf_ex2_train=plot_pacf(df_train['Exogenous 2'], lags=168, title= 'Partial Autocorrelation EX2 - Train')

fig_acf_ex2_train.savefig(r'D:\Datasets\epfmodels\charts\fig_acf_ex2_train.jpg',dpi=300,bbox_inches="tight")
fig_pacf_ex2_train.savefig(r'D:\Datasets\epfmodels\charts\fig_pacf_ex2_train.jpg',dpi=300,bbox_inches="tight")

#get pacf values for variables on train set
pacf_price = sm.tsa.pacf(df_train['Price'], nlags=168)
pacf_ex1 = sm.tsa.pacf(df_train['Exogenous 1'], nlags=168)
pacf_ex2 = sm.tsa.pacf(df_train['Exogenous 2'], nlags=168)

#get acf values for variables on train set
acf_price = sm.tsa.acf(df_train['Price'], nlags=168)
acf_ex1 = sm.tsa.acf(df_train['Exogenous 1'], nlags=168)
acf_ex2 = sm.tsa.acf(df_train['Exogenous 2'], nlags=168)

np.save(r'D:\Datasets\epfmodels\metadata\pacf_price.npy',pacf_price)
np.save(r'D:\Datasets\epfmodels\metadata\pacf_ex1.npy',pacf_ex1)
np.save(r'D:\Datasets\epfmodels\metadata\pacf_ex2.npy',pacf_ex2)

np.save(r'D:\Datasets\epfmodels\metadata\acf_price.npy',acf_price)
np.save(r'D:\Datasets\epfmodels\metadata\acf_ex1.npy',acf_ex1)
np.save(r'D:\Datasets\epfmodels\metadata\acf_ex2.npy',acf_ex2)

print(type(pacf_price))
print(sum(pacf_price))
pacf_scaled = [x/sum(pacf_price) for x in pacf_price]
print(pacf_scaled)
print(sum(pacf_scaled))

#rolling_mean and pacf-based weighted average features
df_train['Pricerolling_mean'] = df_train['Price'].rolling(window=24).mean()
df_train['Exogenous 1rolling_mean'] = df_train['Exogenous 1'].rolling(window=24).mean()
df_train['Exogenous 2rolling_mean'] = df_train['Exogenous 2'].rolling(window=24).mean()

df_train['weighted_price'] = 0.0
print(df_train['weighted_price'].head())
print(df_train['Price'].shape[0])
for j in range(df_train['Price'].shape[0]):
    weighted_combination=0
    if math.isnan(df_train['Pricelag_24'].values[j])==False:
        for m in range(24):
            lagpos=m+1
            weighted_combination = weighted_combination+df_train['Pricelag_'+str(lagpos)].values[j]*pacf_scaled[m]
        #print('Weighted Combination of row', j)    
        #print(weighted_combination)
        df_train['weighted_price'].values[j]=weighted_combination

#print(df_train['lag_'+str(1)])
    

#expanding window feature
df_train['Priceexpanding_mean'] = df_train['Price'].expanding(24).mean()
df_train['Exogenous 1expanding_mean'] = df_train['Exogenous 1'].expanding(24).mean()
df_train['Exogenous 2expanding_mean'] = df_train['Exogenous 2'].expanding(24).mean()

print(df_train.shape)
#Remove rows that have NaN
df_train = df_train.dropna()

####FORMAT TEST SET ACCORDINGLY
#Date features - TEST
df_test['Date'] = pd.to_datetime(df_test.index,format='%d-%m-%Y %H:%M')
df_test['year'] = df_test['Date'].dt.year
df_test['month'] = df_test['Date'].dt.month
df_test['day'] = df_test['Date'].dt.day
df_test['dayofweek_num']=df_test['Date'].dt.dayofweek
df_test['dayofweek_name']=df_test['Date'].dt.day_name()

df_test['Hour'] = df_test['Date'].dt.hour
df_test['Quarter'] = df_test['Date'].dt.quarter

#Create Lag Features for the previous day -TEST
df_test = lagfeatures(df_test, featurename='Price', nlags=24)
df_test = lagfeatures(df_test, featurename='Exogenous 1', nlags=24)
df_test = lagfeatures(df_test, featurename='Exogenous 2', nlags=24)

plot_acf(df_test['Price'], lags=168, title= 'Autocorrelation Price - Test')
plot_pacf(df_test['Price'], lags=168, title= 'Partial Autocorrelation Price - Test')

#check correlation for exogenous variables -test
plot_acf(df_test['Exogenous 1'], lags=168, title= 'Autocorrelation EX1 - Test')
plot_pacf(df_test['Exogenous 1'], lags=168, title= 'Partial Autocorrelation EX1 - Test')

plot_acf(df_test['Exogenous 2'], lags=168, title= 'Autocorrelation EX2 - Test')
plot_pacf(df_test['Exogenous 2'], lags=168, title= 'Partial Autocorrelation EX2 - Test')

#get pacf values for price -test
pacf_test = sm.tsa.pacf(df_test['Price'], nlags=168)

print(pacf_test)
print(sum(pacf_test))
pacf_scaled_test = [x/sum(pacf_test) for x in pacf_test]
print(pacf_scaled_test)
print(sum(pacf_scaled_test))

#rolling_mean and pacf-based weighted average features - test
df_test['Pricerolling_mean'] = df_test['Price'].rolling(window=24).mean()
df_test['Exogenous 1rolling_mean'] = df_test['Exogenous 1'].rolling(window=24).mean()
df_test['Exogenous 2rolling_mean'] = df_test['Exogenous 2'].rolling(window=24).mean()

df_test['weighted_price'] = 0.0
print(df_test['weighted_price'].head())
print(df_test['Price'].shape[0])
for j in range(df_test['Price'].shape[0]):
    weighted_combination=0
    if math.isnan(df_test['Pricelag_24'].values[j])==False:
        for m in range(24):
            lagpos=m+1
            weighted_combination = weighted_combination+df_test['Pricelag_'+str(lagpos)].values[j]*pacf_scaled_test[m]
        #print('Weighted Combination of row', j)    
        #print(weighted_combination)
        df_test['weighted_price'].values[j]=weighted_combination

#print(df_train['lag_'+str(1)])
    

#expanding window feature - test
df_test['Priceexpanding_mean'] = df_test['Price'].expanding(24).mean()
df_test['Exogenous 1expanding_mean'] = df_test['Exogenous 1'].expanding(24).mean()
df_test['Exogenous 2expanding_mean'] = df_test['Exogenous 2'].expanding(24).mean()

print(df_test.shape)
#Remove rows that have NaN - test
df_test = df_test.dropna()


 
#Save and print training dataset after feature engineering
df_train.to_csv(r'D:\Datasets\epfmodels\NP_train.csv',index=False)
df_test.to_csv(r'D:\Datasets\epfmodels\NP_test.csv',index=False)

#df_train.to_excel(r'D:\Datasets\epfmodels\NP_train_xl.xlsx')
print(df_train.dtypes)
#print(tabulate(df_train.head(),headers='keys'))

print(df_train.shape)
print(df_test.shape)
#print(tabulate(df_train.iloc[[25]],headers='keys'))


"""
#DEFINE INPUT FOR TRAIN AND TEST SET VALUES
trainset = df_train.values
testset = df_test.values



print(trainset.shape)
print(trainset[:5])
print(testset.shape)

trainset_daily, testset_daily=split_daily(trainset, testset)

print(trainset_daily.shape)
print(trainset_daily[:5])
print(testset_daily.shape)

#Build the training inputs and outputs
trainset_X, trainset_Y=supervised_conversion(trainset_daily, n_input=24, n_out=24)
testset_X, testset_Y=supervised_conversion(testset_daily, n_input=24, n_out=24)


print(trainset_X[:1])
print(trainset_Y[:1])
print(trainset_X.shape)
"""

"""

#RUN DNN EXAMPLE
dataset='NP'
nlayers = 2
years_test = 2
begin_test_date = None
end_test_date = None
shuffle_train = 1
data_augmentation = 0
new_hyperopt = 1
calibration_window = 4
experiment_id = 1
max_evals = 5
"""



"""
path_datasets_folder = r'D:\Datasets\epfmodels\datasets'
path_hyperparameters_folder = r'D:\Datasets\epfmodels\experimental_files'

hyperparameter_optimizer(path_datasets_folder=path_datasets_folder, 
                         path_hyperparameters_folder=path_hyperparameters_folder, 
                         new_hyperopt=new_hyperopt, max_evals=max_evals, nlayers=nlayers, dataset=dataset, 
                         years_test=years_test, calibration_window=calibration_window, 
                         shuffle_train=shuffle_train, data_augmentation=0, experiment_id=experiment_id,
                         begin_test_date=begin_test_date, end_test_date=end_test_date)
"""

#DNN MODEL TEST FOR PREDICTION
"""
neurons=[64,32]
model=DNNModel(neurons=neurons,n_features=2,outputShape=24,dropout=0,batch_normalization=0,
               lr=None,verbose=True,epochs_early_stopping=40,scaler=None,loss='mae',
               optimizer='adam',activation='relu',initializer='glorot_uniform',regularization=None,lambda_reg=0)

model.clear_session()
model.fit(X_train,Y_train,X_test,Y_test)
ypred=model.predict(X_test)
print(ypred)
"""

