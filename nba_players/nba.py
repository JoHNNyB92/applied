#!/usr/bin/python

import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn import decomposition
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from collections import Counter
import operator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import sys
import csv
import numpy as np
from scipy import stats


def main(argv):

 print 'Number of arguments:', len(sys.argv), 'arguments.'
 print 'Argument List:', str(sys.argv)


 dataset_folder= 'noisy_datasets'
 nba= pd.read_excel(dataset_folder + '/'+ sys.argv[1])
 nba_test = pd.read_excel(dataset_folder + '/'+ sys.argv[2])
 nba_clean= pd.read_excel(dataset_folder + '/'+ 'players_stats_0.05__train_clean.xlsx')
 #nba = pd.read_excel('players_stats_0.05_f_train_noisy.xlsx')
 #nba = pd.read_excel('players_stats_0.15_c_train_noisy.xlsx')
 #nba_test = pd.read_excel('players_stats_0.05__test_.xlsx')

 '''
 print "\nHead of the dataset:\n", nba.head()
 print "------------------------------------------------------------------------------------------------"
 print "\nTail of the dataset:\n", nba.tail()
 '''

 null_data = nba[nba.isnull().any(axis=1)]
 #print null_data

 null_data = nba_test[nba_test.isnull().any(axis=1)]
 #print null_data

 #train
 features = nba
 features = features.drop('SALARY', axis=1)
 features = features.drop('Player', axis=1)
 labels = nba['SALARY'].values
 features = features.values
 print "\nShape of nba train:", features.shape

 #test
 features_test = nba_test
 features_test = features_test.drop('SALARY', axis=1)
 features_test = features_test.drop('Player', axis=1)
 labels_test = nba_test['SALARY'].values
 features_test = features_test.values
 print "\nShape of nba test:", features_test.shape


 #train_clean
 features_c = nba_clean
 features_c = features_c.drop('SALARY', axis=1)
 features_c = features_c.drop('Player', axis=1)
 labels_c= nba_clean['SALARY'].values
 features_c = features_c.values
 print "\nShape of nba train:", features_c.shape

 '''
 print (nba.describe())
 pd.set_option('display.width', 100)
 pd.set_option('precision', 3)
 correlations = nba.corr(method='pearson')

 #correlation matrix
 print(correlations)
 corr_matrix = nba.corr().abs()
 sns.heatmap(correlations)
 plt.show()

 #histograms
 h = nba.hist(figsize=(20,10))
 plt.show()

 #density
 nba.plot(kind='density', subplots=True, layout=(6,5), sharex=False, figsize=(50,50))
 plt.show()

 #boxplot
 sns.boxplot(data=nba)
 plt.show()
 '''
 # Rescale data (between 0 and 1)

 #train
 scaler = MinMaxScaler(feature_range=(0, 1))
 rescaledX = scaler.fit_transform(features)
 scaler2=MinMaxScaler()
 rescaledY=scaler2.fit_transform(labels.reshape(-1, 1))
 # summarise transformed data

 #test
 scaler_test = MinMaxScaler(feature_range=(0, 1))
 rescaledX_test = scaler_test.fit_transform(features_test)
 scaler2_test=MinMaxScaler()
 rescaledY_test=scaler2_test.fit_transform(labels_test.reshape(-1, 1))
 # summarise transformed data
 
 #train clean
 scaler_c = MinMaxScaler(feature_range=(0, 1))
 rescaledX_c = scaler_c.fit_transform(features_c)
 scaler2_c=MinMaxScaler()
 rescaledY_c=scaler2_c.fit_transform(labels_c.reshape(-1, 1))
 # summarise transformed data

 #Standardise Data
 #train
 scaler = StandardScaler().fit(rescaledX)
 standardX = scaler.transform(rescaledX)
 scaler2=StandardScaler().fit(rescaledY)
 standardY=scaler2.transform(rescaledY)
 '''
 print "Mean after scalling for train is:", standardX.mean()
 print "Variance after scalling for train is:", standardX.var()
 '''

 #Standardise Data
 #test
 scaler_test = StandardScaler().fit(rescaledX_test)
 standardX_test = scaler_test.transform(rescaledX_test)
 scaler2_test=StandardScaler().fit(rescaledY_test)
 standardY_test=scaler2_test.transform(rescaledY_test)
 '''
 print "Mean after scalling for test is:", standardX_test.mean()
 print "Variance after scalling for test is:", standardX_test.var()
 '''


 #train clean
 scaler_c = StandardScaler().fit(rescaledX_c)
 standardX_c = scaler_c.transform(rescaledX_c)
 scaler2_c=StandardScaler().fit(rescaledY_c)
 standardY_c=scaler2_c.transform(rescaledY_c)

 #train
 # Normalise data (length of 1)
 scaler = Normalizer().fit(standardX)
 normalizedX = scaler.transform(standardX)
 scaler2 = Normalizer().fit(standardY)
 normalizedY = scaler2.transform(standardY)

 X=normalizedX
 Y=normalizedY


 #test
 # Normalise data (length of 1)
 scaler_test = Normalizer().fit(standardX_test)
 normalizedX_test = scaler_test.transform(standardX_test)
 scaler2_test = Normalizer().fit(standardY_test)
 normalizedY_test = scaler2_test.transform(standardY_test)

 X_test=normalizedX_test
 Y_test=normalizedY_test
 


 #train clean
 # Normalise data (length of 1)
 scaler_c = Normalizer().fit(standardX_c)
 normalizedX_c = scaler_c.transform(standardX_c)
 scaler2_c = Normalizer().fit(standardY_c)
 normalizedY_c= scaler2_c.transform(standardY_c)

 X_c=normalizedX_c
 Y_c=normalizedY_c




 seed=7
 kfold = KFold(n_splits=10, random_state=seed)
 
 '''CLEAN'''
 
 '''NOISY'''

 #linear_test
 print "----------------------------------------------linear regression--------------------------------------------------"
  #linear_model
 model_LR_clean = LinearRegression()
 model_LR_clean.fit(X_c,Y_c)
 predictions_LR_c=model_LR_clean.predict(X_test)
 lin_mse_c = mean_squared_error(predictions_LR_c,Y_test)
 lin_rmse_c = np.sqrt(lin_mse_c)
 lin_var_c=predictions_LR_c.var()
 print('CLEAN:Linear Regression RMSE: %.4f' % lin_rmse_c)
 print('CLEAN:Linear Regression var: %.4f' % lin_var_c)


 #lasso_test
 print "----------------------------------------------lasso regression--------------------------------------------------"
  #Lasso
 model_Lasso_clean =Lasso()
 model_Lasso_clean.fit(X_c,Y_c)
 predictions_Lasso_c=model_Lasso_clean.predict(X_test)
 lasso_mse_c = mean_squared_error(predictions_Lasso_c,Y_test)
 lasso_mse_c = np.sqrt(lasso_mse_c)
 lasso_var_c=predictions_Lasso_c.var()
 print('CLEAN:Lasso Regression RMSE: %.4f' % lasso_mse_c)
 print('CLEAN:Lasso Regression var: %.4f' % lasso_var_c)

 #ridge_test
 print "----------------------------------------------ridge regression-------------------------------------------------"
  #Ridge
 model_Ridge_clean =Ridge()
 model_Ridge_clean.fit(X,Y)
 predictions_Ridge_c=model_Ridge_clean.predict(X_test)
 Ridge_mse_c = mean_squared_error(predictions_Ridge_c,Y_test)
 Ridge_mse_c = np.sqrt(Ridge_mse_c)
 ridge_var_c=predictions_Ridge_c.var()
 print('CLEAN:Ridge Regression RMSE: %.4f' % Ridge_mse_c)
 print('CLEAN:Ridge Regression var: %.4f' % ridge_var_c)
 
 
 

 '''NOISY'''

 #linear_test
 print "----------------------------------------------linear regression--------------------------------------------------"
  #linear_model
 model_LR_win = LinearRegression()
 model_LR_win.fit(X,Y)
 predictions_LR=model_LR_win.predict(X_test)
 lin_mse = mean_squared_error(predictions_LR,Y_test)
 lin_rmse = np.sqrt(lin_mse)
 lin_var=predictions_LR.var()
 print('NOISY:Linear Regression RMSE: %.4f' % lin_rmse)
 print('NOISY:Linear Regression var: %.4f' % lin_var)


 #lasso_test
 print "----------------------------------------------lasso regression--------------------------------------------------"
  #Lasso
 model_Lasso_win =Lasso()
 model_Lasso_win.fit(X,Y)
 predictions_Lasso=model_Lasso_win.predict(X_test)
 lasso_mse = mean_squared_error(predictions_Lasso,Y_test)
 lasso_mse = np.sqrt(lasso_mse)
 lasso_var=predictions_Lasso.var()
 print('NOISY:Lasso Regression RMSE: %.4f' % lasso_mse)
 print('NOISY:Lasso Regression var: %.4f' % lasso_var)

 #ridge_test
 print "----------------------------------------------ridge regression-------------------------------------------------"
  #Ridge
 model_Ridge_win =Ridge()
 model_Ridge_win.fit(X,Y)
 predictions_Ridge=model_Ridge_win.predict(X_test)
 Ridge_mse = mean_squared_error(predictions_Ridge,Y_test)
 Ridge_mse = np.sqrt(Ridge_mse)
 ridge_var=predictions_Ridge.var()
 print('NOISY:Ridge Regression RMSE: %.4f' % Ridge_mse)
 print('NOISY:Ridge Regression var: %.4f' % ridge_var)

 dict={}
 dict['LR']=predictions_LR
 dict['Lasso']=predictions_Lasso
 dict['Ridge']=predictions_Ridge

 dict['LR_c']=predictions_LR_c
 dict['Lasso_c']=predictions_Lasso_c
 dict['Ridge_C']=predictions_Ridge_c

 print("###############################################################")


 '''T-test'''
 print("CLEAN LINEAR - NOISY LINEAR")
 t2, p2 = stats.ttest_ind(predictions_LR_c,predictions_LR)
 print("t = " + str(t2))
 print("p = " + str(2*p2))

 print("CLEAN LASSO - NOISY LASSO")
 t2, p2 = stats.ttest_ind(predictions_Lasso_c,predictions_Lasso)
 print("t = " + str(t2))
 print("p = " + str(2*p2))
 
 print("CLEAN RIDGE - NOISY RIDGE")
 t2, p2 = stats.ttest_ind(predictions_Ridge_c,predictions_Ridge)
 print("t = " + str(t2))
 print("p = " + str(2*p2))

if __name__ == "__main__":
   main(sys.argv[1:])
    

