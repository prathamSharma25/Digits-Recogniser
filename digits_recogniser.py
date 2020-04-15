# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:16:48 2020

@author: Pratham
"""

"""
Digits Recogniser - Machine Learning model to predict hand-drawn digit

This model identifies the hand-drawn digit based on data about each pixel of 
the image of the hand-drawn digit. Digits from 0 through 9 are hand-drawn. 

Training data consists of 785 columns. 
The first column, called "label", is the digit that was drawn by the user. 
The rest of the columns contain the pixel-values of the associated image. 

Test data consists of 784 columns containing pixel-values of the image.

Clearly, this is a classification problem .
Logistic Regression algorithm is used here to train the ML model to predict 
the digit drawn. 
Hence, data is classified into 10 classes for each digit from 0 through 9.

Output of the model is stored as a csv file with 2 columns. The first column
denotes the "ImageID" for each image in the test data, ranging from 1 to 28000.
Second column ("Label") contains the digit as predicted by the model.  
"""


# Importing required libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Loading the training dataset and identifying independent and dependent
# features (X_train and y_train, respectively)
digits_dataset = pd.read_csv("train.csv")
X_train = digits_dataset.iloc[:, 1:].values
y_train = digits_dataset.iloc[:, 0].values


# Loading the testing dataset (X_test)
X_test = pd.read_csv("test.csv")


# Performng feature scaling for X_train as well as X_test using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Training a logistic regression ML model
from sklearn.linear_model import LogisticRegression
digit_classifier = LogisticRegression(random_state = 0)
digit_classifier.fit(X_train, y_train)


# Performing predictions on testing data
y_pred = digit_classifier.predict(X_test)


# Storing predicted values ("Label") along with "ImageID" as csv file
imageid = list(range(1, 28001))
data = {"ImageID" : imageid, "Label" : y_pred}
results = pd.DataFrame(data)
results.to_csv("submission.csv", index = False)