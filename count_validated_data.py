from collections import Counter 
import numpy as np
import pandas as pd
import csv
import scipy.stats
import timeit
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn import metrics #for mean absolute error/mean squared error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sys import argv
import warnings

warnings.filterwarnings('always')

#print("\nProblem 1\n")
file = "crime_10_2018.csv"
data = pd.read_csv(file)

#print(data.shape)

# create a Python list of feature names
feature_cols = ['OFFENSE_CODE', 'YEAR', 'MONTH']
#Select X and y for regression
# use the list to select a subset of the original DataFrame
X = data[feature_cols]

codes = {}
for row in X.values:
    if (row[0]) in codes.keys():
        codes[row[0]] = codes[row[0]] + 1
    else:
        codes[row[0]] = 1

print("Total validated in 2018: " + str(sum(codes.values())))
actual_total = sum(codes.values())
actual_percent = {}
for entry in codes.keys():
    print(codes[entry], entry)
    actual_percent[entry] = (float(codes[entry])/actual_total)*100
actual_percent = sorted((value, key) for (key,value) in actual_percent.items())
actual_percent.reverse()
#sorted_d = sorted((value, key) for (key,value) in codes.items())
#sorted_d.reverse()