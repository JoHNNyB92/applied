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



#nba = pd.read_excel('players_stats_0.05__train_clean.xlsx')
#nba = pd.read_excel('players_stats_0.05_f_train_noisy.xlsx')
nba = pd.read_excel('players_stats_0.05_c_train_noisy.xlsx')
nba_test = pd.read_excel('players_stats_0.05__test_.xlsx')

print "\nHead of the dataset:\n", nba.head()
print "------------------------------------------------------------------------------------------------"
print "\nTail of the dataset:\n", nba.tail()

null_data = nba[nba.isnull().any(axis=1)]
print null_data

null_data = nba_test[nba_test.isnull().any(axis=1)]
print null_data

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
print (nba.describe())


'''
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

#Standardise Data
#train
scaler = StandardScaler().fit(rescaledX)
standardX = scaler.transform(rescaledX)

scaler2=StandardScaler().fit(rescaledY)
standardY=scaler2.transform(rescaledY)
print "Mean after scalling for train is:", standardX.mean()
print "Variance after scalling for train is:", standardX.var()


#Standardise Data
#test
scaler_test = StandardScaler().fit(rescaledX_test)
standardX_test = scaler_test.transform(rescaledX_test)

scaler2_test=StandardScaler().fit(rescaledY_test)
standardY_test=scaler2.transform(rescaledY_test)
print "Mean after scalling for test is:", standardX_test.mean()
print "Variance after scalling for test is:", standardX_test.var()

#train
# Normalise data (length of 1)
scaler = Normalizer().fit(standardX)
normalizedX = scaler.transform(standardX)


scaler2 = Normalizer().fit(standardY)
normalizedY = scaler.transform(standardY)

X=normalizedX
Y=normalizedY


#test
# Normalise data (length of 1)
scaler_test = Normalizer().fit(standardX_test)
normalizedX_test = scaler.transform(standardX_test)


scaler2_test = Normalizer().fit(standardY_test)
normalizedY_test = scaler_test.transform(standardY_test)

X_test=normalizedX_test
Y_test=normalizedY_test

seed=7
kfold = KFold(n_splits=10, random_state=seed)



models = []
models.append(('LR',      LinearRegression()))
models.append(('RIDGE',   Ridge()))
models.append(('LASSO',   Lasso(alpha=0.1)))


# The scoring function to use
scoring = ['neg_mean_squared_error','neg_mean_absolute_error','r2']

# We are going to evaluate all classifiers, and store results in two lists:
results_mse={}
results_mae={}
results_r2={}
results = {}
names   = []
for score in scoring:
    results={}
    names   = []
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=score)
        cv_results=(cv_results)
        results[name]=cv_results.mean()
        names.append(name)
        msg = "%s: %f (%f)--%s" % (name, cv_results.mean(), cv_results.std(),score)
        print(msg)
    if score=='neg_mean_squared_error':
        results_mse=results
    elif score=='neg_mean_absolute_error':
        results_mae=results
    else:
        results_r2=results
        
        
        
results_r2_sorted=  sorted(results_r2.items(), key=operator.itemgetter(1))
results_mse_sorted=  sorted(results_mse.items(), key=operator.itemgetter(1))
results_mae_sorted=  sorted(results_mae.items(), key=operator.itemgetter(1))
print "R^2:\n", results_r2_sorted
print "MSE:\n", results_mse_sorted
print "MAE:\n", results_mae_sorted


print "1st choice on training:",(results_mae_sorted[2])
print "2nd choice on training:",(results_mae_sorted[1])


regressors=['LR', 'RIDGE','LASSO']

#With Hyper Parameters Tuning
for i in regressors:
    if i=="LR":
        model_LR=  LinearRegression()
        params_LR = {
            'fit_intercept':[True,False],
            'normalize':[True,False],
            'copy_X':[True, False] }
    elif i=="RIDGE":
        model_Ridge=Ridge()
        params_Ridge ={'solver' : ['svd','lsqr' ,'auto','cholesky'],
                 'alpha':[0.001,0.01,0.1,0.2],
                 'fit_intercept':[True, False],
                 'normalize':[True,False],
                 'tol':[0.001, 0.002, 0.01,0.1]}
    else: 
        model_Lasso=Lasso()
        params_Lasso = { 'alpha':[0.001,0.01,0.1,0.2],
                    'fit_intercept':[True, False],
                   'normalize':[True,False],
                  'max_iter':[1000,2000,500],
                    'tol':[0.001, 0.002, 0.01,0.1],
                 'selection':['cyclic','random']}



#linear_model

model_LR_win = GridSearchCV(model_LR, param_grid=params_LR, n_jobs=-1)
#Learning
model_LR_win.fit(X,Y)
#The best hyper parameters set
print("Best Hyper Parameters:",model_LR_win.best_params_)
#Prediction
prediction=model_LR_win.predict(X_test)

score='neg_mean_absolute_error'

neg_mean_absolute_error = cross_val_score(model_LR, X, Y, cv=kfold, scoring='neg_mean_absolute_error')
neg_mean_absolute_error=neg_mean_absolute_error.mean()
print "neg_mean_absolute_error after setting hyperparameters is:",(neg_mean_absolute_error)

#Lasso
model_Lasso_win = GridSearchCV(model_Lasso, param_grid=params_Lasso, n_jobs=-1)
#Learning
model_Lasso_win.fit(X,Y)
#The best hyper parameters set
print("Best Hyper Parameters:",model_Lasso_win.best_params_)
#Prediction
prediction=model_Lasso_win.predict(X_test)
score='neg_mean_absolute_error'
neg_mean_absolute_error = cross_val_score(model_Lasso, X, Y, cv=kfold, scoring='neg_mean_absolute_error')
neg_mean_absolute_error=neg_mean_absolute_error.mean()
print "neg_mean_absolute_error after setting hyperparameters is:",(neg_mean_absolute_error)

#Ridge
model_Ridge_win = GridSearchCV(model_Ridge, param_grid=params_Ridge, n_jobs=-1)
#Learning
model_Ridge_win.fit(X,Y)
#The best hyper parameters set
print("Best Hyper Parameters:",model_Lasso_win.best_params_)
#Prediction
prediction=model_Ridge_win.predict(X_test)
#importing the metrics module
score='neg_mean_absolute_error'
#evaluation(Accuracy)
neg_mean_absolute_error = cross_val_score(model_Ridge, X, Y, cv=kfold, scoring='neg_mean_absolute_error')
neg_mean_absolute_error=neg_mean_absolute_error.mean()
print "neg_mean_absolute_error after setting hyperparameters is:",(neg_mean_absolute_error)

#linear_test

model_LR_win.fit(X_test,Y_test)
predictions_LR=model_LR_win.predict(X_test)
neg_mean_absolute_error_LR = cross_val_score(model_LR_win, X_test, Y_test, cv=kfold, scoring='neg_mean_absolute_error')
neg_mean_absolute_error_LR=neg_mean_absolute_error_LR.mean()
print "The model is :", model_LR_win
print "neg_mean_absolute_error for test dataset after setting  hyperparameters is:",(neg_mean_absolute_error_LR)


#lasso_test

model_Lasso_win.fit(X_test,Y_test)
predictions_Lasso=model_Lasso_win.predict(X_test)
neg_mean_absolute_error_Lasso = cross_val_score(model_Lasso_win, X_test, Y_test, cv=kfold, scoring='neg_mean_absolute_error')
neg_mean_absolute_error_Lasso=neg_mean_absolute_error_Lasso.mean()
print "The  model is :", model_Lasso_win
print "neg_mean_absolute_error for test dataset after setting  hyperparameters is:",(neg_mean_absolute_error_Lasso)

#ridge_test
model_Ridge_win.fit(X_test,Y_test)
predictions_Ridge=model_Ridge_win.predict(X_test)
neg_mean_absolute_error_Ridge = cross_val_score(model_Ridge_win, X_test, Y_test, cv=kfold, scoring='neg_mean_absolute_error')
neg_mean_absolute_error_Ridge=neg_mean_absolute_error_Ridge.mean()
print "The  model is :", model_Ridge_win
print "neg_mean_absolute_error for test dataset after setting  hyperparameters is:",(neg_mean_absolute_error_Ridge)
