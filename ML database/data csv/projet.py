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
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.tree import DecisionTreeRegressor
# load the data set 
data = pd.read_csv('data csv.csv') 

# print info about columns in the dataframe 
print(data.info()) 
#what kinds of data it contains
print(data.keys())
#a description of what the features are:
#print(data.DESCR)
# creating dummies for GAME
dummies_GAME = pd.get_dummies(data.GAME)

dummies_GAME.columns=['DICE','DICE2','FREE','FREE2']

rows = dummies_GAME.shape[0]
i = 0
while i < rows:
     if dummies_GAME['DICE'][i] < dummies_GAME['DICE2'][i]:
             dummies_GAME['DICE'][i]=dummies_GAME['DICE2'][i] 
     i+=1
    
i = 0
while i < rows:
     if dummies_GAME['FREE'][i] < dummies_GAME['FREE2'][i]:
             dummies_GAME['FREE'][i]=dummies_GAME['FREE2'][i] 
     i+=1     
data = pd.concat([data,dummies_GAME],axis='columns')
data = data.drop(['GAME','DICE2','FREE2'],axis='columns')

# creating dummies for BET
dummies_BET = pd.get_dummies(data.BET)

dummies_BET.columns=['HI','HI1','LO','LO1','LO2','LO3']
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
    if dummies_BET['LO'][i] < dummies_BET['LO3'][i]:
        dummies_BET['LO'][i]=dummies_BET['LO3'][i]
    i+=1
    
data = pd.concat([data,dummies_BET],axis='columns')
data = data.drop(['BET','HI1','LO1','LO2','LO3'],axis='columns')

X = data[['DICE','FREE','HI','LO','ROLL','STAKE','MULT']]
y = data['PROFIT']
########################################################################################################################################
# Multivariate Linear Regression steps:
#step1
# def initialize_parameters(lenw):
#     w = np.random.randn(1,lenw)
#     # w = np.zeros((1,lenw))
#     b = 0
#     return w,b
# #step2
# def forward_prop(X,w,b):  # w--> 1xn     ,    X --> nxm
#     z = np.dot(w,X)+b  #  z --> 1xm b_vector = [b b b b ....]
#     return z
# #step3
# def cost_function(z,y):
#     m = y.shape[1]
#     J = (1/(2*m))*np.sum(np.square(z-y))
#     return J
# #step4
# def back_prop(X,y,z):
#     m = y.shape[1]
#     dz = (1/m)*(z-y)
#     dw = np.dot(dz,X,T)    #dw --> 1xn
#     db = np.sum(dz)
#     return dw,db
# #step5
# def gradient_descent_update(w,b,dw,db,learning_rate):
#     w = w - learning_rate*dw
#     b = b - learning_rate*db
#     return w,b
# #step6
# def linear_regression_model(X_train,y_train,X_val,y_val,learning_rate,epochs):
#
#     lenw = X_train.shape[0]
#     w,b = initialize_parameters(lenw)  #step1
#
#     cost_train = []
#     m_train = y_train.shape[1]
#     m_val = y_val.shape[1]
#
#     for i in range(1,epochs+1):
#         z_train = forward_prop(X_train,w,b)  #step2
#         cost_train = cost_function(z_train,y_train)  #step3
#         dw,db = back_prop(X_train,y_train,z_train)  #step4
#         w,b = gradient_descent_update(w,b,dw,db,learning_rate)  #step5
#
#         #store training cost in a list for plotting purpose
#         if i%10 ==0:
#             cost_train.append(cost_train)
#         #MAE_train
#         MAE_train = (1/m_train)*np.sum(np.abs(z_train-y_train))
#
#         #cost_val, MAE_val
#         z_val = forward_prop(X_val,w,b)
#         cost_val = cost_function(z_val,y_val)
#         MAE_val = (1/m_val)*np.sum(np.abs(z_val-y_val))
#
#         #print out cost_train,cost_val,MAE_train,MAE_val
#
#         print('Epochs '+str(i)+'/'+str(epochs)+': ')
#         print('Training cost '+str(cost_train)+'|'+'Validation cost '+str(cost_val))
#         print('Training MEA '+str(cost_train)+'|'+'Validation cost '+str(cost_val))
# plt.plot(cost_train)
# plt.xlabel('Iterations(per tens)')
# plt.ylabel('Training cost')
# plt.title('Learning rate '+str(learning_rate))
# plt.show()
         
# regression TRY:
print('khmira center go:\n')
#data['PROFIT','HI','LO'] = data.target 
#ycenter=pd.DataFrame(np.c_[data['PROFIT'],data['HI'],data['LO']], columns=['PROFIT','HI','LO'])
#Xcenter=pd.DataFrame(np.c_[data['ROLL'],data['DICE'],data['FREE'],data['STAKE'],data['MULT']], columns=['ROLL','DICE','FREE','STAKE','MULT'])
Xcenter=data['ROLL','DICE','FREE','STAKE','MULT']
Ycenter=data['PROFIT','HI','LO'] 
   # split into 70:30 ration 
X_traincenter, X_testcenter, y_traincenter, y_testcenter = train_test_split(Xcenter, ycenter, test_size = 0.3, random_state = 0) 
# describes info about traincenter and testcenter set 
print("Number transactions X_traincenter dataset: ", X_traincenter.shape) 
print("Number transactions y_traincenter dataset: ", y_traincenter.shape) 
print("Number transactions X_testcenter dataset: ", X_testcenter.shape) 
print("Number transactions y_testcenter dataset: ", y_testcenter.shape)

print("Before OverSampling, counts of positive profitcenter: {}".format(sum(y_traincenter > 0))) 
print("Before OverSampling, counts of negative profitcenter: {} \n".format(sum(y_traincenter < 0))) 
 #Initialize the linear model center
regcenter = LinearRegression()

 #Train the model with our training datacenter
regcenter.fit(X_traincenter,y_traincenter)

 #Print the coeffiscients/weights for each feature/column of our model
print(regcenter.coef_) #f(x,a) = mx + da + b =y

 #Print the predictions on our test datacenter
y_predcenter = regcenter.predict(X_testcenter)
print(y_predcenter)

 #Print the actual values center
print(y_testcenter)

 #Check the model performance/accuracy using Mean Squared Error (MSE)
print(np.mean((y_predcenter - y_testcenter)**2))

 #Check the model performance/accuracy using Mean Squared Error (MSE) and sklearn.metrics
print(mean_squared_error(y_testcenter,y_predcenter))
 #check the predictions against the actual values by using the RMSE and R-2 metrics
test_setcenter_rmse = (np.sqrt(mean_squared_error(y_testcenter, y_predcenter)))
print('test_setcenter_rmse = ',test_setcenter_rmse)
test_setcenter_r2 = r2_score(y_testcenter, y_predcenter)
print('test_setcenter_r2 = ',test_setcenter_r2)
 #########################################################################################
print('End Of Center')
 #################### End Of Center ######################################################
 #########################################################################################
# split into 70:30 ration 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 

# describes info about train and test set 
#print("Number transactions X_train dataset: ", X_train.shape) 
#print("Number transactions y_train dataset: ", y_train.shape) 
#print("Number transactions X_test dataset: ", X_test.shape) 
#print("Number transactions y_test dataset: ", y_test.shape) 

#print("Before OverSampling, counts of positive profit: {}".format(sum(y_train > 0))) 
#print("Before OverSampling, counts of negative profit: {} \n".format(sum(y_train < 0))) 


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
print("khmira go")
#Visualize the profit data
#plt.figure(figsize=(16,8))
#plt.title('ROLL HISTORY')
#plt.xlabel('Moment')
#plt.ylabel('Profit (BTC)')
#plt.plot(data['PROFIT'])
#plt.show()
# #Initialize the linear model
# reg = linear_model.LinearRegression()

 #Train the model with our training data
 #reg.fit(X_train,y_train)

 #Print the coeffiscients/weights for each feature/column of our model
# print(reg.coef_) #f(x,a) = mx + da + b =y

 #Print the predictions on our test data
 #y_pred = reg.predict(X_test)
 #print(y_pred)

 #Print the actual values
 #print(y_test)

 #Check the model performance/accuracy using Mean Squared Error (MSE)
 #print(np.mean((y_pred - y_test)**2))

 #Check the model performance/accuracy using Mean Squared Error (MSE) and sklearn.metrics
 #print(mean_squared_error(y_test,y_pred))