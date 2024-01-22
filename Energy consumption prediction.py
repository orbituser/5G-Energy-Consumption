# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:14:51 2024

@author: mozea
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# from buildml.automate import SupervisedLearning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import metrics

dataset = pd.read_csv("5G_energy_consumption_dataset.csv")
data = pd.read_csv("5G_energy_consumption_dataset.csv")

# Initial EDA
data.info()
data_more_desc_statistic = data.describe(include = "all")
data_mode = data.mode()
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr() 
data_total_null_count = data.isnull().sum().sum()
data_hist = data.hist(figsize = (15, 10), bins = 10)


# Data Cleaning and Transformation
# Converting time column to date time format
data["Time"] = pd.to_datetime(data["Time"])

# Extracting Date Features for Time Series Analysis 
data["Year"] = data["Time"].dt.year
data["Month"] = data["Time"].dt.month
data["Day"] = data["Time"].dt.day
data["Hour"] = data["Time"].dt.hour

# Drop Time Column
data = data.drop("Time", axis = 1)

# Drop Duplicate Columns
data = data.drop_duplicates()

# Transform BS Column
data = pd.get_dummies(data, drop_first = True, dtype = np.int8)


# Second EDA(Optional)
# data.info()
# data_more_desc_statistic = data.describe(include = "all")
# data_mode = data.mode()
# data_distinct_count = data.nunique()
# data_correlation_matrix = data.corr() 
# data_total_null_count = data.isnull().sum().sum()
# data_hist = data.hist(figsize = (15, 10), bins = 10)


# Further Data Preparation and Segregation
x = data.drop("Energy", axis = 1)
y = data.Energy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 55)

# Using BuildML
# regressors = [
#     LinearRegression(),
#     DecisionTreeRegressor(random_state = 55),
#     RandomForestRegressor(random_state = 55),
#     KNeighborsRegressor(),
#     GaussianNB(),
#     BernoulliNB(),
#     SVR(),
#     XGBRegressor(random_state = 55)
#     ]

# Model building
regressor = RandomForestRegressor()
model = regressor.fit(x_train, y_train)

# Model Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)

# Model Evalaution
r2 = r2_score(y_test, y_pred1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred1))












