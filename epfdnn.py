# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:57:35 2021

@author: dimkonto
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from sklearn import preprocessing
import scipy
from tabulate import tabulate
from matplotlib import pyplot as pp
import datetime
import statistics
import math
from math import log,sqrt
from scipy import stats
from mlxtend.evaluate import bias_variance_decomp

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD,Adam
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import soft_dtw

import seaborn as sns
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from keras.callbacks import TensorBoard

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import product
from scipy.optimize import differential_evolution
from dtw import dtw,accelerated_dtw
from scipy import signal




def test_causality(Xcauses,Yeffect):
        maxlag=2
        test = 'params_ftest'
        causaltempdf=pd.DataFrame()    
        causaltempdf['1']=Yeffect
        causaltempdf['2']=Xcauses
        #causaltempdf=causaltempdf.fillna(0)
        print(causaltempdf)
        try:
            gc_res=grangercausalitytests(causaltempdf, maxlag=maxlag, verbose=True)
            p_values = [round(gc_res[i+1][0][test][0],4) for i in range(maxlag)]
            result=np.min(p_values)
            return result
        except:
            return 5

#LINEAR REGRESSION IMPORTANCE
def feature_importance(input_data,output_data,output_dimension):
    """
    #linear regression importance
    lr_model = LinearRegression()
    # fit the model
    for i in range(output_dimension):
        lr_model.fit(input_data, output_data[:,i])
        # get importance
        lr_importance = lr_model.coef_
        np.save(r'D:\Datasets\epfmodels\metadata\lrimportance_hour_'+str(i)+'.npy',lr_importance)
        #print(type(lr_importance))
        for j,v in enumerate(lr_importance):
            print('Feature: %0d, Score: %.5f' % (j,v))
    
        # plot feature importance
        pp.bar([x for x in range(len(lr_importance))], lr_importance)
        pp.title("LR Importance For Hour: " + str(i+1))
        pp.show()
        
    #dtr importance
    dtr_model = DecisionTreeRegressor()
    # fit the model
    for i in range(output_dimension):
        dtr_model.fit(input_data, output_data[:,i])
        # get importance
        dtr_importance = dtr_model.feature_importances_
        np.save(r'D:\Datasets\epfmodels\metadata\dtrimportance_hour_'+str(i)+'.npy',dtr_importance)
        #print(type(dtr_importance))
        
        for j,v in enumerate(dtr_importance):
            print('Feature: %0d, Score: %.5f' % (j,v))
    
        # plot feature importance
        pp.bar([x for x in range(len(dtr_importance))], dtr_importance)
        pp.title("DTR Importance For Hour: " + str(i+1))
        pp.show()        
    
    
    #rf importance
    rf_model = RandomForestRegressor(n_estimators=5,max_depth=241)
    # fit the model
    for i in range(output_dimension):
        rf_model.fit(input_data, output_data[:,i])
        # get importance
        rf_importance = rf_model.feature_importances_
        np.save(r'D:\Datasets\epfmodels\metadata\rfimportance_hour_'+str(i)+'.npy',rf_importance)
        #print(type(rf_importance))
        
        for j,v in enumerate(rf_importance):
            print('Feature: %0d, Score: %.5f' % (j,v))
    
        # plot feature importance
        pp.bar([x for x in range(len(rf_importance))], rf_importance)
        pp.title("RF Importance For Hour: " + str(i+1))
        pp.show()        
    
    """
    
    #XGBoost Importance
    #xgb_model = XGBRegressor()
    # fit the model
    #for i in range(output_dimension):
    #    xgb_model.fit(input_data, output_data[:,i])
        # get importance
    #    xgb_importance = xgb_model.feature_importances_
    #    np.save(r'D:\Datasets\epfmodels\metadata\xgbimportance_hour_'+str(i)+'.npy',xgb_importance)
        #print(type(rf_importance))
        
    #    for j,v in enumerate(xgb_importance):
    #       print('Feature: %0d, Score: %.5f' % (j,v))
    
        # plot feature importance
    #    pp.bar([x for x in range(len(xgb_importance))], xgb_importance)
    #    pp.title("XGB Importance For Hour: " + str(i+1))
    #    pp.show()        
    
    """
    #Ridge Permutation Importance
    rp_model = Ridge() #KNeighborsRegressor(n_neighbors=2, algorithm='brute')
    # fit the model
    for i in range(output_dimension):
        rp_model.fit(input_data, output_data[:,i])
        # get importance
        result_perm = permutation_importance(rp_model,input_data, output_data[:,i], scoring='neg_mean_squared_error')
        
        #print(type(rf_importance))
        rp_importance = result_perm.importances_mean
        np.save(r'D:\Datasets\epfmodels\metadata\ridgeimportance_hour_'+str(i)+'.npy',rp_importance)
        for j,v in enumerate(rp_importance):
            print('Feature: %0d, Score: %.5f' % (j,v))
    
        # plot feature importance
        pp.bar([x for x in range(len(rp_importance))], rp_importance)
        pp.title("Ridge Importance For Hour: " + str(i+1))
        pp.show()
    """  
    
    
    #SAVE RESULTS TO NP ARRAYS BECAUSE IT IS SLOW    
    """
    #ENCODE OUTPUT LABELS FIRST BASED ON THE RANGE OF PRICES (STEP-WISE) 
    #logistic regression importance
    log_model = LogisticRegression()
    # fit the model
    for i in range(output_dimension):
        log_model.fit(input_data, output_data[:,i])
        # get importance
        log_importance = log_model.coef_[0]
        # summarize feature importance
        for j,v in enumerate(log_importance):
            print('Feature: %0d, Score: %.5f' % (j,v))
        # plot feature importance
        pp.bar([x for x in range(len(log_importance))], log_importance)
        pp.show()
    """

#BUILD FEATURE SELECTOR
def feature_selector():
    methodlist=['dtr','lr','rf','ridge','xgb']
    top_features=np.full(shape=(5,24,10), fill_value=np.nan)
    
    
    for m in range(len(methodlist)):
        
        for h in range(24):
            imp_h = np.load(r'D:\Datasets\epfmodels\metadata\\'+str(methodlist[m])+'importance_hour_'+str(h)+'.npy')
            print(imp_h)
            ind = np.argpartition(imp_h, -10)[-10:]
            top_features[m][h]=ind
            print(ind)
            print(imp_h[ind])
            #for j,v in enumerate(imp_h):
            #    print('Feature: %0d, Score: %.5f' % (j,v))
            # plot feature importance
            pp.bar([x for x in range(len(imp_h))], imp_h)
            pp.title(methodlist[m]+" Importance For Hour: " + str(h+1))
            pp.show()
            #break
        #break
    print(top_features[0])
    np.save(r'D:\Datasets\epfmodels\metadata\topfeatures.npy',top_features)

def feature_ranking(flag):
    topf=np.load(r'D:\Datasets\epfmodels\metadata\topfeatures.npy')
    methodlist=['dtr','lr','rf','ridge','xgb']
    
    print(topf[0])
    
    """
    # STRATEGY #1: KEEPING FEATURES THAT AFFECT MOST HOURS LEADS TO :(COMPRESSED FEATURE SET, NO CONTRIBUTION FOR EVERY HOUR)
    dictlist = [dict() for x in range(topf.shape[0])]
    for i in range(topf.shape[0]):
        #count_dict={}
        dictlist[i]={}
        for j in range(topf.shape[1]):
            for k in range(topf.shape[2]):
                candidate_value = topf[i][j][k]
                candidate_index = [i,j,k]
                count = 0
                for m in range(topf.shape[1]):
                    if candidate_value in topf[i][m]: #search occurencies of candidate in the 24h of that method
                        count = count + 1
                print(candidate_value, count)
                #count_dict[candidate_value] = count
                dictlist[i][candidate_value]=count
        break
    #print(count_dict)
    #print(count_dict[236])
    print(dictlist[0])
    """
    if flag==-1:
        # STRATEGY #2: KEEPING THE TOP FEATURES FOR EACH HOUR (JUST BY DUPLICATE ELIMINATION ON EACH METHOD)
        #FLATTEN FOR EACH METHOD, MERGE AND REMOVE DUPLICATES
        flat_topf=topf.flatten()
        print(flat_topf.shape)
        featurelist = np.unique(flat_topf)
        print(featurelist.shape)
    else:
        # STRATEGY #3: BASED ON A FLAG, SELECT ONLY THE FEATURE LIST FROM ONE METHOD
        print(topf[0], topf[0].shape)
        
        flat_topf=topf[flag].flatten()
        print(flat_topf.shape)
        featurelist = np.unique(flat_topf)
        print(featurelist.shape)
    
    
    return featurelist


#BUILD DNN(MLP) MODEL
def DNN_epf(input_dimension,neurons_h1, neurons_h2, neurons_out):
    nn=Sequential()
    nn.add(Dense(neurons_h1,activation='relu',input_dim=input_dimension)) #first hidden layer feeding from input
    nn.add(Dense(neurons_h2)) #second hidden layer
    nn.add(Dense(neurons_out)) #output
    
    #APPLYING TIME-BASED LEARNING RATE DECAY FOR SGD (OTHERWISE DEFAULT OPTIMIZER IS ADAM)
    epochs = 4000
    learning_rate = 0.0005
    decay_rate = 1e-6
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    
    nn.compile(loss='mae',optimizer=sgd)
    plot_model(nn, to_file='D:\Datasets\epfmodels\model.png')
    return nn

def LSTM_epf(in_seq_len, out_seq_len, n_features, units):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(in_seq_len, n_features)))
    #model.add(LSTM(units, activation='relu'))
    model.add(Dense(out_seq_len))
    
    #APPLYING TIME-BASED LEARNING RATE DECAY FOR SGD (OTHERWISE DEFAULT OPTIMIZER IS ADAM)
    epochs = 4000
    learning_rate = 0.0005
    decay_rate = 1e-6
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    optimizer = Adam(lr=learning_rate)
    
    model.compile(optimizer=sgd, loss='mae')
    return model

def calculate_mape(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def model_evaluation(actual_inverted, predicted_inverted):
    mape_dnn = calculate_mape(actual_inverted, predicted_inverted)
    mse_dnn = mean_squared_error(actual_inverted, predicted_inverted)
    rmse_dnn = math.sqrt(mse_dnn)
    mae_dnn = mean_absolute_error(actual_inverted, predicted_inverted)

    print(mape_dnn, mse_dnn, rmse_dnn, mae_dnn)
    return mape_dnn, mse_dnn, rmse_dnn, mae_dnn

def DNN_run(trainset_X,trainset_Y,testset_X,testset_Y, n1, n2, nout, bsize, eps, name):
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
    
    n_X_train=input_scaler.fit_transform(trainset_X)
    n_Y_train=output_scaler.fit_transform(trainset_Y)

    n_X_test=input_scaler.fit_transform(testset_X)
    n_Y_test=output_scaler.fit_transform(testset_Y)
    
    print(n_X_train.shape)
    
    nn=DNN_epf(n_X_train.shape[1], n1, n2, nout)
    
    #EARLY STOPPING
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=100)
    bestmodelpath = r'D:\Datasets\epfmodels\dnn_'+name+'.h5'
    mc = ModelCheckpoint(bestmodelpath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    
    history=nn.fit(n_X_train,n_Y_train,batch_size=bsize,validation_data=(n_X_test,n_Y_test),epochs=eps, verbose=2, callbacks=[es, mc])

    testPredict = nn.predict(n_X_test)

    pp.plot(history.history['loss'], label='train')
    pp.plot(history.history['val_loss'], label='test')
    pp.legend()
    pp.savefig(r'D:\Datasets\epfmodels\charts\dnn_loss_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()

    print(n_Y_test)
    print(testPredict)
    actual_inverted = output_scaler.inverse_transform(n_Y_test)
    predicted_inverted = output_scaler.inverse_transform(testPredict)
    model_evaluation(actual_inverted, predicted_inverted)
    
    #RETURN ERROR TERMS
    prediction_train = nn.predict(n_X_train)
    actual_train_inv = output_scaler.inverse_transform(n_Y_train)
    predicted_train_inv = output_scaler.inverse_transform(prediction_train)
    
    error_train = actual_train_inv - predicted_train_inv
    error_test = actual_inverted - predicted_inverted

    print(error_test)
    
    if name == 'BC-MLP-PREDICTION':
        return predicted_inverted
    else:
        return error_train,error_test, predicted_inverted, actual_inverted

def LSTM_run(trainset_X,trainset_Y,testset_X,testset_Y, units, bsize, eps, name):
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
    
    n_X_train=input_scaler.fit_transform(trainset_X)
    n_Y_train=output_scaler.fit_transform(trainset_Y)

    n_X_test=input_scaler.fit_transform(testset_X)
    n_Y_test=output_scaler.fit_transform(testset_Y)
    
    print(n_X_train.shape)
    
    #3D input for LSTM [samples, timesteps, features]
    samples = n_X_train.shape[0]
    timesteps = n_X_train.shape[1]
    features = 1
    n_X_train = n_X_train.reshape((samples,timesteps, features))
    n_X_test = n_X_test.reshape((n_X_test.shape[0],timesteps,features))
    
    print(n_X_train)
    
    nn=LSTM_epf(n_X_train.shape[1], n_Y_train.shape[1], features, 24)
    
    #EARLY STOPPING
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=100)
    bestmodelpath = r'D:\Datasets\epfmodels\lstm_'+name+'.h5'
    mc = ModelCheckpoint(bestmodelpath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    
    history=nn.fit(n_X_train,n_Y_train,batch_size=bsize,validation_data=(n_X_test,n_Y_test),epochs=eps, verbose=2, callbacks=[es, mc])

    testPredict = nn.predict(n_X_test)

    pp.plot(history.history['loss'], label='train')
    pp.plot(history.history['val_loss'], label='test')
    pp.legend()
    pp.savefig(r'D:\Datasets\epfmodels\charts\lstm_loss_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()

    print(n_Y_test)
    print(testPredict)
    actual_inverted = output_scaler.inverse_transform(n_Y_test)
    predicted_inverted = output_scaler.inverse_transform(testPredict)
    model_evaluation(actual_inverted, predicted_inverted)
    

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def calculate_info_metrics(model, input_X, output_Y):
    yhat = model.predict(input_X.reshape(-1, 1))
    mse = mean_squared_error(output_Y, yhat)
    num_params = len(model.coef_) + 1
    n = input_X.shape[0]
    
    aic = n * log(mse) + 2 * num_params
    bic = n * log(mse) + num_params * log(n)
    print(mse,aic,bic)
    return aic

def error_evaluation(trainset_X,trainset_Y, errorset, testset_X, testset_Y, price_prediction, price_actual, name):
    print(trainset_X.shape, errorset.shape)
    error_corr=np.full(shape=(trainset_X.shape[1],errorset.shape[1]), fill_value=np.nan)
    error_p=np.full(shape=(trainset_X.shape[1],errorset.shape[1]), fill_value=np.nan)
    
    data_corr=np.full(shape=(trainset_X.shape[1],trainset_Y.shape[1]), fill_value=np.nan)
    data_p=np.full(shape=(trainset_X.shape[1],trainset_Y.shape[1]), fill_value=np.nan)
    
    
    
    for i in range(errorset.shape[1]):
        for j in range(trainset_X.shape[1]):
            r, p = stats.pearsonr(trainset_X[:,j],errorset[:,i])
            rdata, pdata = stats.pearsonr(trainset_X[:,j],trainset_Y[:,i])
            error_corr[j][i] = r
            error_p[j][i] = p
            data_corr[j][i] = rdata
            data_p[j][i] = pdata
            print(r,p)
            #break
        #break
    print(error_corr[:,0].shape)
    print(error_p[:,0].shape)
    
    #PLOTTING ERROR CORRELATION TO FEATURES
    err_corr_flist=[]
    data_corr_flist=[]
    for i in range(errorset.shape[1]):
        top_err_corrs = np.argpartition(abs(error_corr[:,i]), -10)[-10:]
        err_corr_flist.append(top_err_corrs.tolist())
        
        pp.plot(error_corr[:,i], label='%s corr_err' % i)
        #pp.title("Ridge Importance For Hour: " + str(i+1))
        #break
    pp.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    pp.show()
    #PLOTTING OUTPUT CORRELATION TO FEATURES
    for i in range(trainset_Y.shape[1]):
        top_data_corrs = np.argpartition(abs(data_corr[:,i]), -10)[-10:]
        data_corr_flist.append(top_data_corrs.tolist())
        
        pp.plot(data_corr[:,i], label='%s corr_data' % i)
        #pp.title("Ridge Importance For Hour: " + str(i+1))
        #break
    pp.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    pp.show()
    
    err_corr_flist = [item for sublist in err_corr_flist for item in sublist]
    err_corr_flist = list(dict.fromkeys(err_corr_flist))
    
    data_corr_flist = [item for sublist in data_corr_flist for item in sublist]
    data_corr_flist = list(dict.fromkeys(data_corr_flist))
    
    ### LINEAR REGRESSION MODEL TO SEE WHICH FEATURES CAN PREDICT ERRORS BASED ON INFORMATION THEORY METRICS
    #for i in range(trainset_X.shape[1]):
        #res = AutoReg(endog=errorset[:,0],exog=trainset_X[:,i], lags=0, old_names=True).fit()
        #print(res.aic, res.hqic, res.bic)
    for j in range(errorset.shape[1]):    
        aiclist=[]
        for i in range(trainset_X.shape[1]):
            lr_ins = LinearRegression()
            lr_ins.fit(trainset_X[:,i].reshape(-1, 1), errorset[:,j])
            
            ##BIAS VARIANCE
            #avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            #    lr_ins, trainset_X[:,i].reshape(-1, 1), errorset[:,j], trainset_X[:,i].reshape(-1, 1), errorset[:,j], 
            #    loss='mse',
            #    num_rounds=100,
            #    random_seed=123)


            #print('Average expected loss: %.3f' % avg_expected_loss)
            #print('Average bias: %.3f' % avg_bias)
            #print('Average variance: %.3f' % avg_var)
            ##
            
            aicout=calculate_info_metrics(lr_ins,trainset_X[:,i],errorset[:,j])
            aiclist.append(aicout)
        pp.plot(aiclist)
        #pp.savefig(r'D:\Datasets\epfmodels\charts\aic_hour_'+str(j)+'.jpg',dpi=300,bbox_inches="tight")
        pp.show()
    
    #ROLLING CORRELATION OF MOST GENERALLY CORRELATED FEATURE FROM ABOVE (SINCE IT IS COMPUTATIONALLY EXPENSIVE)
    rolling_error_corr=np.full(shape=(trainset_X.shape[0]-23,errorset.shape[1]), fill_value=np.nan)
    rolling_error_p=np.full(shape=(trainset_X.shape[0]-23,errorset.shape[1]), fill_value=np.nan)
    
    """
    for i in range(errorset.shape[1]):
        rw_error=rolling_window(errorset[:,i],24)
        rw_feature=rolling_window(trainset_X[:,j],24)
        for k in range(rw_feature.shape[0]):
            rwin, pwin = stats.pearsonr(rw_error[k],rw_feature[k])
            rolling_error_corr[k][i]=rwin
            rolling_error_p[k][i]=pwin
    
    np.save(r'D:\Datasets\epfmodels\metadata\rolling_error_corr.npy', rolling_error_corr)
    np.save(r'D:\Datasets\epfmodels\metadata\rolling_error_p.npy', rolling_error_p)     
    """       
    rw=rolling_window(errorset[:,0],2*365*24)
    
    #print(rw.shape, errorset[:,0].shape)
    #print(rw[0])
    #print(stats.pearsonr(rw[0],trainset_X[:,0]))
    
    """
    #CHECK TLCC, IPS
    # CALCULATING CROSS-CORRELATION FOR INPUT-ERROR AND INPUT-OUTPUT FEATURES
    signal_err_corr=np.full(shape=(errorset.shape[1], trainset_X.shape[1], trainset_X.shape[0]+errorset.shape[0]-1), fill_value=np.nan)
    signal_feat_corr=np.full(shape=(errorset.shape[1], trainset_X.shape[1], trainset_X.shape[0]+errorset.shape[0]-1), fill_value=np.nan)
    #corr = signal.correlate(errorset[:,0], trainset_X[:,0])
    #print(trainset_X[:,0].shape,corr.shape)
    
    for i in range(errorset.shape[1]):
        for j in range(trainset_X.shape[1]):
            corr = signal.correlate(errorset[:,i], trainset_X[:,j])
            feat_corr = signal.correlate(trainset_Y[:,i], trainset_X[:,j])
            #lags = signal.correlation_lags(len(errorset[:,i]), len(trainset_X[:,j))
            corr /= np.max(corr)
            feat_corr /= np.max(feat_corr)
            
            signal_err_corr[i][j]=corr
            signal_feat_corr[i][j]=feat_corr
            
            #print(corr)
    np.save(r'D:\Datasets\epfmodels\metadata\signal_err_corr.npy', signal_err_corr)
    np.save(r'D:\Datasets\epfmodels\metadata\signal_feat_corr.npy', signal_feat_corr)
    
    pp.plot(signal_err_corr[0][0])
    pp.show()
    
    pp.plot(signal_feat_corr[0][0])
    pp.show()
    """
    ### AUTOREGRESSION MODEL FOR ERROR CORRECTION
    ### mape_dnn, mse_dnn, rmse_dnn, mae_dnn
    mape_store=np.full(shape=(24,2), fill_value=np.nan)
    mse_store=np.full(shape=(24,2), fill_value=np.nan)
    rmse_store=np.full(shape=(24,2), fill_value=np.nan)
    mae_store=np.full(shape=(24,2), fill_value=np.nan)
    
    for h in range(24):
        window = 24
        train_resid=errorset[:,h]
        est_model = AutoReg(errorset[:,h], lags=window, old_names=False)
        est_model_fit = est_model.fit()
        coef = est_model_fit.params
        hvar = train_resid[len(train_resid)-window:]
        hvar = [hvar[i] for i in range(len(hvar))]
        predictions=np.full(shape=testset_Y[:,h].shape, fill_value=np.nan)
        for t in range(len(testset_Y[:,h])):
            # persistence
            yhat = price_prediction[t,h]
            error = testset_Y[t,h] - yhat
            #predict error
            length = len(hvar)
            lag = [hvar[i] for i in range(length-window,length)]
            pred_error = coef[0]
            for d in range(window):
                pred_error += coef[d+1] * lag[window-d-1]
                # correct the prediction (forecast + estimated error)
            yhat = yhat + pred_error
            predictions[t]=yhat
            hvar.append(error)
            #print('predicted=%f, expected=%f' % (yhat, testset_Y[t,h]))
                # error
        rmse = sqrt(mean_squared_error(testset_Y[:,h], predictions))
        #print('Test RMSE: %.3f' % rmse)
        # plot predicted error
        pp.plot(testset_Y[:,h])
        pp.plot(predictions, color='red')
        pp.show()
    
        print(predictions.shape, testset_Y[:,h].shape)
        print('HOUR '+str(h)+': ')
        mape_og, mse_og, rmse_og, mae_og = model_evaluation(price_actual[:,h], price_prediction[:,h])
        mape_erc, mse_erc, rmse_erc, mae_erc = model_evaluation(price_actual[:,h], predictions)
        
        mape_store[h][0]=mape_og
        mse_store[h][0]=mse_og
        rmse_store[h][0]=rmse_og
        mae_store[h][0]=mae_og
        
        mape_store[h][1]=mape_erc
        mse_store[h][1]=mse_erc
        rmse_store[h][1]=rmse_erc
        mae_store[h][1]=mae_erc
        
        print('Average mape, mse, rmse, mae for base: ',np.average(mape_store[:,0]),np.average(mse_store[:,0]),np.average(rmse_store[:,0]),np.average(mae_store[:,0]))
        print('Average mape, mse, rmse, mae for erc: ',np.average(mape_store[:,1]),np.average(mse_store[:,1]),np.average(rmse_store[:,1]),np.average(mae_store[:,1]))
    
    """    
    xax=np.arange(24)    
    pp.bar(xax, mape_store[:,0], label='MAPE Base')
    pp.bar(xax, mape_store[:,1], label='MAPE ERC')
    pp.xlabel('Hours')
    pp.ylabel('Mean Absolute Percentage Error')
    pp.legend(loc = 'best')
    pp.savefig(r'D:\Datasets\epfmodels\charts\dnn_mape_comparison_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    pp.bar(xax, mse_store[:,0], label='MSE Base')
    pp.bar(xax, mse_store[:,1], label='MSE ERC')
    pp.xlabel('Hours')
    pp.ylabel('Mean Squared Error')
    pp.legend(loc = 'best')
    pp.savefig(r'D:\Datasets\epfmodels\charts\dnn_mse_comparison_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    pp.bar(xax, rmse_store[:,0], label='RMSE Base')
    pp.bar(xax, rmse_store[:,1], label='RMSE ERC')
    pp.xlabel('Hours')
    pp.ylabel('Root Mean Squared Error')
    pp.legend(loc = 'best')
    pp.savefig(r'D:\Datasets\epfmodels\charts\dnn_rmse_comparison_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    pp.bar(xax, mae_store[:,0], label='MAE Base')
    pp.bar(xax, mae_store[:,1], label='MAE ERC')
    pp.xlabel('Hours')
    pp.ylabel('Mean Absolute Error')
    pp.legend(loc = 'best')
    pp.savefig(r'D:\Datasets\epfmodels\charts\dnn_mae_comparison_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    pp.plot(predictions)
    #pp.savefig(r'D:\Datasets\epfmodels\charts\dnn_hourly_demo prediction_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    """
    
    ### STATIONARITY AND AUTOCORRELATION EVALUATION OF ERROR - FINAL CODE BLOCK
    """
    adf_stat=[]
    xaxis_val=['1%', '5%', '10%']
    acf_hours=[]
    for s in range(24):
        plot_acf(errorset[:,s], lags=48, title= 'Residual Error Autocorrelation for Hour '+str(s))
        acf_err_vals = sm.tsa.acf(errorset[:,s], nlags=24)
        acf_hours.append(acf_err_vals)
        pp.savefig(r'D:\Datasets\epfmodels\charts\error_adfcutoff_'+str(s)+name+'.jpg',dpi=300,bbox_inches="tight")
        pp.show()
        crit_values=[]
        stationarity=adfuller(errorset[:,s])
        print('ADF for error of hour '+str(s))
        print(stationarity)
        adf_stat.append(stationarity[0])
        for key,value in stationarity[4].items():
            crit_values.append(value)
        pp.plot(crit_values)
    pp.xlabel('Critical cutoff')
    pp.ylabel('Cutoff values')
    pp.xticks(range(len(xaxis_val)), xaxis_val)
    #pp.savefig(r'D:\Datasets\epfmodels\charts\error_adfcutoff_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    pp.plot(adf_stat)
    pp.xlabel('Hour')
    pp.ylabel('Augmented Dickey Fuller Statistic')
    #pp.savefig(r'D:\Datasets\epfmodels\charts\error_adfstatistic_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    
    avg_acf_list=[]
    for s in range(24):
        point_vec=[]
        for n in range(24):
            print(acf_hours[n][s])
            point_vec.append(acf_hours[n][s])
        avg_acf_list.append(sum(point_vec)/len(point_vec))
    
    print(avg_acf_list)
    pp.plot(avg_acf_list)
    pp.xlabel('Hour')
    pp.ylabel('Average Lag Autocorrelation')
    #pp.savefig(r'D:\Datasets\epfmodels\charts\avg_errorlag_acf_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    """
    
    ######TESTING - INVESTIGATE PLOT AIC FOR EACH HOUR FOR ERROR ESTIMATION (NOT FINAL CODE BLOCK)
    
    aic_all=[]
    bic_all=[]
    hqic_all=[]
    for h in range(24):
        train_resid=errorset[:,h]
        acf_err_vals = sm.tsa.acf(train_resid, nlags=48)
        dict_acf= { i : acf_err_vals[i] for i in range(0, len(acf_err_vals) ) }
        print(dict_acf)
        out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'
        aiclist=[]
        biclist=[]
        hqiclist=[]
    
        est_model = AutoReg(errorset[:,h], lags=24, old_names=False).fit()
        #print(out.format(est_model.aic, est_model.hqic, est_model.bic))
        aiclist.append(round(est_model.aic,4))
        biclist.append(round(est_model.bic,4))
        hqiclist.append(round(est_model.hqic,4))
        X1=['AIC 0.2', 'A']
        for l in range(3):
            dlist = list((k) for k, v in dict_acf.items() if v >= 0.2+l*0.1)
            ddict = dict((k,v) for k, v in dict_acf.items() if v >= 0.2+l*0.1)
            #print(ddict)
            #print(dlist)
    
            est_model = AutoReg(errorset[:,h], lags=dlist[1:], old_names=False).fit()
            #print(out.format(est_model.aic, est_model.hqic, est_model.bic))
            aiclist.append(round(est_model.aic,4))
            biclist.append(round(est_model.bic,4))
            hqiclist.append(round(est_model.hqic,4))
        aic_all.append(aiclist)
        bic_all.append(biclist)
        hqic_all.append(hqiclist)
    print(aic_all)
    
    """
    ### AIC TABLE   
    colors_all = []
    for h in range(24):
        color_hour=[]
        for m in range(4):
            if aic_all[h][0] <aic_all[h][m]:
                color_hour.append("r")
            if aic_all[h][0] >aic_all[h][m]:
                color_hour.append("b")
            if aic_all[h][0] ==aic_all[h][m]:
                color_hour.append("w")
        colors_all.append(color_hour)
    
    print(colors_all)
    
    columns = ['24 Lag Window','ACF>=0.2','ACF>=0.3','ACF>=0.4']
    rows=[]
    for h in range(24):
        rows.append("AIC H"+str(h))
    
    fig, ax = pp.subplots()
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=aic_all,cellColours=colors_all,
                         colLabels=columns,rowLabels=rows,loc='center')
    #pp.savefig(r'D:\Datasets\epfmodels\charts\AICTABLE_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()      
    """
    """
    ### BIC TABLE
    colors_all = []
    for h in range(24):
        color_hour=[]
        for m in range(4):
            if bic_all[h][0] <bic_all[h][m]:
                color_hour.append("r")
            if bic_all[h][0] >bic_all[h][m]:
                color_hour.append("b")
            if bic_all[h][0] ==bic_all[h][m]:
                color_hour.append("w")
        colors_all.append(color_hour)
    
    print(colors_all)
    
    columns = ['24 Lag Window','ACF>=0.2','ACF>=0.3','ACF>=0.4']
    rows=[]
    for h in range(24):
        rows.append("BIC H"+str(h))
    
    fig, ax = pp.subplots()
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=aic_all,cellColours=colors_all,
                         colLabels=columns,rowLabels=rows,loc='center')
    pp.savefig(r'D:\Datasets\epfmodels\charts\BICTABLE_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()      
    """
    
    ### HQIC TABLE
    """
    colors_all = []
    for h in range(24):
        color_hour=[]
        for m in range(4):
            if hqic_all[h][0] <hqic_all[h][m]:
                color_hour.append("r")
            if hqic_all[h][0] >hqic_all[h][m]:
                color_hour.append("b")
            if hqic_all[h][0] ==hqic_all[h][m]:
                color_hour.append("w")
        colors_all.append(color_hour)
    
    print(colors_all)
    
    columns = ['24 Lag Window','ACF>=0.2','ACF>=0.3','ACF>=0.4']
    rows=[]
    for h in range(24):
        rows.append("HQIC H"+str(h))
    
    fig, ax = pp.subplots()
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=aic_all,cellColours=colors_all,
                         colLabels=columns,rowLabels=rows,loc='center')
    pp.savefig(r'D:\Datasets\epfmodels\charts\HQICTABLE_'+name+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
    
    
    print('Average mape, mse, rmse, mae for base: ',np.average(mape_store[:,0]),np.average(mse_store[:,0]),np.average(rmse_store[:,0]),np.average(mae_store[:,0]))
    print('Average mape, mse, rmse, mae for erc: ',np.average(mape_store[:,1]),np.average(mse_store[:,1]),np.average(rmse_store[:,1]),np.average(mae_store[:,1]))
    """
    
    for s in range(24):
        plot_pacf(errorset[:,s], lags=162, title= 'Residual Error Partial Autocorrelation for Hour '+str(s))
        
    return err_corr_flist, data_corr_flist, predictions
    
###MAIN PROJECT BODY
path_train = r'D:\Datasets\epfmodels\NP_train_epf.csv'
path_test = r'D:\Datasets\epfmodels\NP_test_epf.csv'

df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

print(df_train.shape)
print(df_test.shape)

print(df_train.head())

#PREPARE INPUT 

#TRAIN SET OUTPUT INITIALIZATION (24H VECTORS OF PRICE)
#for 3d input
#df_train_X=np.full(shape=(df_train.shape[0]-23,24,241), fill_value=np.nan)

#2D
df_train_Y=np.full(shape=(df_train.shape[0]-23,24), fill_value=np.nan)


#TEST SET OUTPUT INITIALIZATION (24H VECTORS OF PRICE)
#for 3d
#df_test_X=np.full(shape=(df_test.shape[0]-23,24,241), fill_value=np.nan)

#2D
df_test_Y=np.full(shape=(df_test.shape[0]-23,24), fill_value=np.nan)


#CREATE TRAIN SET (3D INTERPRETATION)
#FILL OUTPUT
target_var =  df_train['Price']

df_train = df_train.drop(columns=['Price'])

#print(df_train.values[0])
df_train_np = df_train.to_numpy()

#print(df_train_np[0:24])
#print(df_train_np[0:24].shape)
#for 3d
#df_train_X[0] = df_train_np[0:24]
#print(df_train_X[0])

#CREATING TRAIN OUTPUT
for i in range(df_train_Y.shape[0]):
    #for 3d
    #df_train_X[i]= df_train_np[0+i:24+i]
    for j in range(df_train_Y.shape[1]):
        df_train_Y[i][j]= target_var.values[i+j]
        

#CREATE TEST SET (3D INTERPRETATION)
#FILL TEST OUTPUT
target_var_test =  df_test['Price']

df_test = df_test.drop(columns=['Price'])

#print(df_train.values[0])
df_test_np = df_test.to_numpy()

#print(df_test_np[0:24])
#print(df_test_np[0:24].shape)
#for 3d
#df_test_X[0] = df_test_np[0:24]
#print(df_train_X[0])
for i in range(df_test_Y.shape[0]):
    #for 3d
    #df_test_X[i]= df_test_np[0+i:24+i]
    for j in range(df_test_Y.shape[1]):
        df_test_Y[i][j]= target_var_test.values[i+j]        

#array = df_train_np.flatten()
#indexer = np.arange(24)[None, :] + 2*np.arange(df_train.shape[0]-23)[:, None]
#print(array[indexer])
#print(array[indexer][0])

#TRY SIMPLER INPUT SET
df_train.drop(df_train.tail(23).index,inplace=True)
df_test.drop(df_test.tail(23).index,inplace=True)

#TRAIN AND TEST INPUT DATASETS ARE THE REMAINING FEATURES IN THE DATASET
df_train_X=df_train.values
df_test_X=df_test.values

#INSPECT SHAPES OF INPUT AND OUTPUT FOR TRAIN AND TEST
print("TRAIN X AND SHAPE")
print(df_train_X,df_train_X.shape)
print(df_train_Y,df_train_Y.shape)
#print("TEST X AND SHAPE")
#print(df_test_X,df_test_X.shape)
#print(df_test_Y,df_test_Y.shape)

### ANALYSIS OF FEATURES AND FEATURE IMPORTANCE

##TRY PCA TO COMPRESS DAILY PRICE MEASUREMENTS TO A LOWER NUMBER OF DIMENSIONS
in_scaler = StandardScaler()
out_scaler = StandardScaler()

s_X_train = in_scaler.fit_transform(df_train_X)
s_Y_train = out_scaler.fit_transform(df_train_Y)

#s_X_test = in_scaler.fit_transform(df_test_X)
#s_Y_test = out_scaler.fit_transform(df_test_Y)

print(s_Y_train)
principal_components = 3
pca = PCA(n_components=principal_components)
components = pca.fit_transform(s_Y_train)
print(components)
print(pca.explained_variance_ratio_)

##TRY TO EVALUATE FEATURE IMPORTANCE
#Linear Regression Importance
#feature_importance(df_train_X,df_train_Y,df_train_Y.shape[1])

#LOAD IMPORTANCES AND MANIPULATE ARRAYS
#lrimp_1=np.load(r'D:\Datasets\epfmodels\metadata\lrimportance_hour_'+str(1)+'.npy')
#for j,v in enumerate(lrimp_1):
#    print('Feature: %0d, Score: %.5f' % (j,v))

#feature_selector()
flist=feature_ranking(flag=-1)
flist=flist.astype(int)
print(flist)

#fig =pp.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(components[:,0],components[:,1],components[:,2])
#pp.show()



###NORMALIZE INPUT AND OUTPUT
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

n_X_train=input_scaler.fit_transform(df_train_X[:,flist])
n_Y_train=output_scaler.fit_transform(df_train_Y)

n_X_test=input_scaler.fit_transform(df_test_X[:,flist])
n_Y_test=output_scaler.fit_transform(df_test_Y)

#print(n_X_train,n_X_train.shape)
#print(n_Y_train,n_Y_train.shape)
#print(input_scaler.inverse_transform(n_X_train))


###CREATE SIMPLE DNN WITH 2 HIDDEN LAYERS
#3d input to 2d if needed
#n_input = df_train_X.shape[1]*df_train_X.shape[2]
#df_train_X = df_train_X.reshape((df_train_X.shape[0],n_input))


###CALLING THE MODEL FOR TRAINING
# TRAINING TENSORS: df_train_X, df_train_Y
# TEST TENSORS: df_test_X, df_test_Y

#MODELS WITH ALL FEATURES
#nn=DNN_epf(df_train_X.shape[1],185, 185, 24)

#MODELS WITH TOP FEATURES


#FIT WITH ALL ORIGINAL FEATURES
"""
n_X_train_og=input_scaler.fit_transform(df_train_X)
n_Y_train_og=output_scaler.fit_transform(df_train_Y)

n_X_test_og=input_scaler.fit_transform(df_test_X)
n_Y_test_og=output_scaler.fit_transform(df_test_Y)

nn_og=DNN_epf(df_train_X.shape[1],100, 52, 24)
history_og=nn_og.fit(n_X_train_og,n_Y_train_og,batch_size=72,validation_data=(n_X_test_og,n_Y_test_og),epochs=300, verbose=2)
testPredict_og = nn_og.predict(n_X_test_og)

pp.plot(history_og.history['loss'], label='train')
pp.plot(history_og.history['val_loss'], label='test')
pp.legend()
pp.savefig(r'D:\Datasets\epfmodels\charts\dnn_loss_nbl100og.jpg',dpi=300,bbox_inches="tight")
pp.show()

print(n_Y_test_og)
print(testPredict_og)
actual_inverted_og = output_scaler.inverse_transform(n_Y_test_og)
predicted_inverted_og = output_scaler.inverse_transform(testPredict_og)
model_evaluation(actual_inverted_og, predicted_inverted_og)


###
#FIT WITH HYBRID FEATURE SELECTOR (batch_size=72)
nn=DNN_epf(df_train_X[:,flist].shape[1],100, 52, 24)
history=nn.fit(n_X_train,n_Y_train,batch_size=72,validation_data=(n_X_test,n_Y_test),epochs=300, verbose=2)

testPredict = nn.predict(n_X_test)

pp.plot(history.history['loss'], label='train')
pp.plot(history.history['val_loss'], label='test')
pp.legend()
pp.savefig(r'D:\Datasets\epfmodels\charts\dnn_loss_nfbl100.jpg',dpi=300,bbox_inches="tight")
pp.show()

print(n_Y_test)
print(testPredict)
actual_inverted = output_scaler.inverse_transform(n_Y_test)
predicted_inverted = output_scaler.inverse_transform(testPredict)
model_evaluation(actual_inverted, predicted_inverted)
"""
print(df_train_X[:,flist].shape)

#CHECK STATIONARITY: ALL TRAINING FEATURES ARE STATIONARY
#for i in range(df_train_X.shape[1]):
#        stationarity=adfuller(df_train_X[:,i])
#        print('ADF statistic: %f p-value: %f' % (stationarity[0],stationarity[1]))

#TEST CAUSALITY SINCE ALL TRAINING FEATURES ARE STATIONARY
cres=test_causality(df_train_X[:,0], df_train_Y[:,0])
print(cres)

#### DNN EXPERIMENTS WITH DIFFERENT FEATURE LISTS
#DNN WITH HYBRID FEATURE SELECTION
#flist=feature_ranking(flag=-1)
#flist=flist.astype(int)
#DNN_run(df_train_X[:,flist], df_train_Y,df_test_X[:,flist],df_test_Y, 100, 52, 24, 72, eps=4000, name='nbfl')

#DNN with each method's feature set
#for i in range(5):
#    methodlist=['dtr','lr','rf','ridge','xgb']
#    flist=feature_ranking(flag=i)
#    flist=flist.astype(int)
#    DNN_run(df_train_X[:,flist], df_train_Y,df_test_X[:,flist],df_test_Y, 100, 52, 24, 72, eps=300, name='dnn_'+methodlist[i])

#DNN WITH ALL FEATURES
#DNN_run(df_train_X, df_train_Y,df_test_X,df_test_Y, 100, 52, 24, 72, eps=4000, name='original')

#LSTM WITH ALL FEATURES (SLOW TRAINING)
#LSTM_run(df_train_X, df_train_Y,df_test_X,df_test_Y, 24, 72, eps=4000, name='original')

###SET UP BC-MLP SYSTEM FOR ERROR CORRECTION
error_train,error_test,price_prediction, price_actual = DNN_run(df_train_X, df_train_Y,df_test_X,df_test_Y, 100, 52, 24, 72, eps=10, name='BC-MLP-GETERROR')
#print(error_train, error_test)
print(df_train_X.shape,error_train.shape, error_test.shape)
new_df=np.concatenate((df_train_X,error_train),axis=1)
print(new_df.shape)
print(df_train_X[:,240],error_train[:,0],new_df[:,240],new_df[:,241])

## EVALUTATE FEATURE SET AND SELECT THE MOST INFLUENTIAL FEATURES TOWARDS THE PREDICTION OF ERROR_TRAIN
error_flist, data_flist, processed_forecast=error_evaluation(df_train_X,df_train_Y, error_train, df_test_X, df_test_Y, price_prediction, price_actual, name='10eps')
print(error_flist,data_flist)
#model_evaluation(price_actual[:,0], price_prediction[:,0])
#model_evaluation(price_actual[:,0], processed_forecast)
##
"""
error_prediction=DNN_run(df_train_X[:,error_flist], error_train,df_test_X[:,error_flist],error_test, 100, 52, 24, 72, eps=1000, name='BC-MLP-PREDICTION')

final_prediction = price_prediction + error_prediction

np.savetxt(r'D:\Datasets\epfmodels\epf_actual_price.csv', price_actual, delimiter=",")
np.savetxt(r'D:\Datasets\epfmodels\epf_error_prediction.csv', error_prediction, delimiter=",")
np.savetxt(r'D:\Datasets\epfmodels\price_prediction.csv', price_prediction, delimiter=",")
np.savetxt(r'D:\Datasets\epfmodels\final_prediction.csv', final_prediction, delimiter=",")

pp.plot(price_actual,label='price_actual')
pp.plot(price_prediction, label='predicted')
pp.plot(final_prediction, label = 'predicted-synth')
pp.legend()
pp.show()

print("Price Prediction")
print(price_prediction)
print("Error Prediction")
print(error_prediction)
print("Error Actual")
print(error_test)
print("Price Actual")
print(price_actual)


model_evaluation(price_actual, price_prediction)
model_evaluation(price_actual, final_prediction)
"""
## SIMPLE EXPERIMENT DNN WITH PEARSON CORELLATED FEATURES and experiment with error curves
#error_train_corr,error_test_corr,price_prediction, price_actual=DNN_run(df_train_X[:,data_flist], df_train_Y,df_test_X[:,data_flist],df_test_Y, 100, 52, 24, 72, eps=10, name='BC-MLP-GETERROR')
#pp.plot(price_actual,label='price_actual')
#pp.plot(price_prediction, label='predicted')
#pp.legend()
#pp.show()
#model_evaluation(price_actual, price_prediction)

#pp.plot(error_train[:,0],label='error_original')
#pp.plot(error_train_corr[:,0], label='error_corr')
#pp.legend()
#pp.show()