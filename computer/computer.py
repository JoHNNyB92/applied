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
 nba_test = pd.read_excel(dataset_folder + '/'+ 'computer___test_.xlsx')
 nba_clean= pd.read_excel(dataset_folder + '/'+ 'computer___train_clean.xlsx')


 '''
 print "\nHead of the dataset:\n", nba.head()
 print "------------------------------------------------------------------------------------------------"
 print "\nTail of the dataset:\n", nba.tail()
 '''

 null_data = nba[nba.isnull().any(axis=1)]
 #print null_data

 null_data = nba_test[nba_test.isnull().any(axis=1)]
 #print null_data
 
 
 #drop na vallues
 data = nba.dropna()
 data_test= nba_test.dropna()
 data_clean = nba_clean.dropna()

 #train
 features = data
 features = features.drop('Attribute Information', axis=1)
 
 features = features.drop('ERP', axis=1)
 features = features.drop('Model Name', axis=1)
 labels = data['ERP'].values
 features = features.values
 print "\nShape of nba train:", features.shape

 #test
 features_test = data_test
 features_test = features_test.drop('Attribute Information', axis=1)
 
 features_test = features_test.drop('Model Name', axis=1)
 features_test = features_test.drop('ERP', axis=1)
 labels_test = data_test['ERP'].values
 features_test = features_test.values
 print "\nShape of nba test:", features_test.shape


 #train_clean
 features_c = data_clean
 features_c = features_c.drop('Attribute Information', axis=1)
 features_c = features_c.drop('Model Name', axis=1)
 features_c = features_c.drop('ERP', axis=1)
 
 labels_c= data_clean['ERP'].values
 #features_c = features_c.values
 print "\nShape of nba train:", features_c.shape


 
 X=features
 Y=labels

 X_test=features_test
 Y_test=labels_test

 X_c=features_c
 Y_c=labels_c
 
 
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
 model_Ridge_clean.fit(X_c,Y_c)
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
 t2, p2 = stats.ttest_ind(predictions_LR,predictions_LR_c)
 print("t = " + str(t2))
 print("p = " + str(2*p2))

 print("CLEAN LASSO - NOISY LASSO")
 t2, p2 = stats.ttest_ind(predictions_Lasso_c ,predictions_Lasso)
 print("t = " + str(t2))
 print("p = " + str(2*p2))
 
 print("CLEAN RIDGE - NOISY RIDGE")
 t2, p2 = stats.ttest_ind(predictions_Ridge_c,predictions_Ridge)
 print("t = " + str(t2))
 print("p = " + str(2*p2))

if __name__ == "__main__":
   main(sys.argv[1:])
    

    

