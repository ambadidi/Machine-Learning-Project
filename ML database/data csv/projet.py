# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:28:36 2020

@author: Khmira
"""


# import necessary modules  
import pandas  as pd 
#import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

# load the data set 
data = pd.read_csv('data csv.csv') 

# print info about columns in the dataframe 
print(data.info()) 

# creating dummies for GAME
dummies_GAME = pd.get_dummies(data.GAME)

dummies_GAME.columns=['DICE','DICE2','FREE']

rows = dummies_GAME.shape[0]
i = 0
while i < rows:
     if dummies_GAME['DICE'][i] < dummies_GAME['DICE2'][i]:
             dummies_GAME['DICE'][i]=dummies_GAME['DICE2'][i] 
     i+=1
     
data = pd.concat([data,dummies_GAME],axis='columns')
data = data.drop(['GAME','DICE2'],axis='columns')

# creating dummies for BET
dummies_BET = pd.get_dummies(data.BET)

dummies_BET.columns=['HI','HI1','LO','LO1','LO2']
rows = dummies_BET.shape[0]
i = 0
while i < rows:
    if dummies_BET['HI'][i] < dummies_BET['HI1'][i]:
             dummies_BET['HI'][i]=dummies_BET['HI1'][i] 
    i+=1

i = 0
while i < rows:
    if dummies_BET['LO'][i] < dummies_BET['LO1'][i]:
        dummies_BET['LO'][i]=dummies_BET['LO1'][i]
    if dummies_BET['LO'][i] < dummies_BET['LO2'][i]:
        dummies_BET['LO'][i]=dummies_BET['LO2'][i]
    i+=1
    
data = pd.concat([data,dummies_BET],axis='columns')
data = data.drop(['BET','HI1','LO1','LO2'],axis='columns')

X = data[['DICE','FREE','HI','LO','ROLL','STAKE','MULT']]
y = data['PROFIT']
# split into 70:30 ration 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 

# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 

print("Before OverSampling, counts of positive profit: {}".format(sum(y_train > 0))) 
print("Before OverSampling, counts of negative profit: {} \n".format(sum(y_train < 0))) 


###################################SMOTE if needed########################################
# import SMOTE module from imblearn library                                              #
#sm = SMOTE(random_state = 2)                                                            # 
#X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())                      #
#print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))         #
#print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))      #
#                                                                                        #
#print("After OverSampling, counts of positive profit: {}".format(sum(y_train_res > 0))) #
#print("After OverSampling, counts of label '0': {}".format(sum(y_train_res < 0)))       #
##########################################################################################
