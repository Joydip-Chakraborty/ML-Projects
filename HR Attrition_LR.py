# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 20:59:53 2021

@author: Lord Voldemort
"""

import pandas as pd
import statsmodels.api as sm
import numpy as nm
#import seaborn as sns
hr_data = pd.read_csv("D:/Skill Edge Training/Assignment 1/T7_HR_Data.csv")
hr_data.head(5)
hr_data.tail(5)
hr_data.info()
print("No. of employees in original dataset:" +str(len(hr_data.index)))
#sns.countplot(x="left", data=hr_data)
#sns.countplot(x="left", hue = "salary" data=hr_data)
hr_data.isnull().sum()
Role_Dummy = pd.get_dummies(hr_data["role"],drop_first=True)
Salary_Dummy = pd.get_dummies(hr_data["salary"],drop_first=True)
hr_data = pd.concat([hr_data,Salary_Dummy, Role_Dummy],axis=1)
hr_data.drop(["salary",],axis=1,inplace=True)
hr_data.drop(["role",],axis=1,inplace=True)

x1=hr_data.drop("left",axis=1)
y1=hr_data["left"]

x2= nm.append(arr = nm.ones((14999,1)).astype(int), values=x1, axis=1)
x_opt= x2[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x2[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x2[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x2[:, [0,1,2,3,4,5,6,7,8,9,10,12,13,16,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x2[:, [0,1,2,3,4,5,6,7,8,9,10,12,13,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x2[:, [0,1,2,3,4,5,6,7,8,9,10,12,13,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x2[:, [0,1,2,3,4,5,6,7,8,9,10,12,13]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size= 0.25, random_state=0)


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

predictions = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_BE_test,predictions)

#Accuracy = (2665+314)/(2665+314+216+555) = 79.44%

#Calculating the coefficients:
print(logmodel.coef_)

#Calculating the intercept:
print(logmodel.intercept_)

#slevel, lival, nump, amh, eic, wrc, prm, low, med, rnd, hr, mgmt