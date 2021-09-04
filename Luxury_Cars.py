# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:42:04 2021

@author: Lord Voldemort
"""
import pandas as pd
import statsmodels.api as sm
import numpy as nm

dataset = pd.read_csv('D:/Skill Edge Training/Assignment 2/T6_Luxury_Cars.csv')
dataset.drop(["Model",],axis=1,inplace=True)
dataset.drop(["Make",],axis=1,inplace=True)
dataset.drop(["Origin",],axis=1,inplace=True)

dataset.info()
dataset.isnull().sum()

pd.get_dummies(dataset["Type"])
S_Dummy = pd.get_dummies(dataset["Type"],drop_first=True)
dataset = pd.concat([dataset,S_Dummy],axis=1)

pd.get_dummies(dataset["DriveTrain"])
S_Dummy = pd.get_dummies(dataset["DriveTrain"],drop_first=True)
dataset = pd.concat([dataset,S_Dummy],axis=1)

dataset.drop(["Type",],axis=1,inplace=True)
dataset.drop(["DriveTrain",],axis=1,inplace=True)

X2=dataset

X = dataset.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,12,13]].values
y = dataset.iloc[:,3].values

X = nm.append(arr = nm.ones((426,1)).astype(int), values=X, axis=1)

x_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= X[:,[0,1,2,3,4,5,7,8,9,10,11,12]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= X[:,[0,2,3,4,5,7,8,9,10,11,12]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= X[:,[0,2,3,4,7,8,9,10,11,12]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

#--------------------------Backward Elimination-------------------------------

x_BE= X2.iloc[:,[1,2,4,7,8,9,10,11,12]].values
y_BE= X2.iloc[:,3].values 

from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_BE_train, y_BE_train)

y_pred= regressor.predict(x_BE_test)

from sklearn.metrics import r2_score
r2_score(y_BE_test,y_pred)

#83.67%

print(regressor.coef_)

print(regressor.intercept_)

#Predicting

print(regressor.predict([[4,200,3230,0,1,0,0,0,1]]))

#30.16

#Regression Equation

#Mileage =61.13 + {Cylinders * (-.422) + Horsepower * (-.012) + Weight * (-.002) + SUV * (-22.5) 
#+ Sedan * (-18.3) + Sports * (-19.7) + Truck * (-22.1) + Wagon * (-18.9) + Front * (1.13)}