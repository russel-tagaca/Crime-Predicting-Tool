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

#import csv file

file = "crime_07_2018.csv"
data = pd.read_csv(file)

# create a Python list of feature names
feature_cols = ['OFFENSE_CODE', 'YEAR', 'MONTH']

#Select X and y for regression
# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# select a Series from the DataFrame
y = data['OFFENSE_CODE']

# 10-fold cross validation on a dataset using a Decision Tree Classifier:
model = DecisionTreeClassifier() 
errorArray = []

kf = KFold(n_splits=10, random_state=42, shuffle=False)
i = 1
for train_index, test_index in kf.split(X):
    # print("Fold %s\nTRAIN SET: %s \n\nTEST SET%s \n\n" % (i, train_index, test_index))
    i += 1
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Get analysis score for each fold
    model.fit(X_train, y_train)
    errorArray.append(model.score(X_test, y_test))

#************MODEL EVALUATION with 10-fold cross validation************
print("\nMean squared error of the model:", np.mean(errorArray))
print("\nRoot mean squared error of the model:", np.sqrt(np.mean(errorArray)))

#************MODEL Creations for Naive Bayes, K-Nearest Neighbors and Decision Tree Classifiers************
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)

#Get run times for each model creation for model analytics
print("\nGet Run Time for all model creations - Scalability")
# instantiate models
# fit the model to the training data (learn the coefficients) and make predictions

#Naive Bayes
start = timeit.default_timer()
naive = GaussianNB()
naive.fit(X_train,y_train)
naive_split_predicted = naive.predict(X_test)
stop = timeit.default_timer()
print('Naive Time: ', stop - start)  

#K-Nearest Neighbors
start = timeit.default_timer()
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_split_predicted = knn.predict(X_test)
stop = timeit.default_timer()
print('KNN Time: ', stop - start)  

#Decision Tree
start = timeit.default_timer()
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
decision_tree_split_predicted = decision_tree.predict(X_test)
stop = timeit.default_timer()
print('Decision tree Time: ', stop - start)  


c = Counter(naive_split_predicted)
codes = {}
for row in X.values:
    if (row[0]) in codes.keys():
        codes[row[0]] = codes[row[0]] + 1
    else:
        codes[row[0]] = 1
print()
predict_total = sum(c.values())
print("Total predicted: " + str(sum(c.values())))
pred_percent = {}
for entry in c.keys():
    pred_percent[entry] = (float(c[entry])/predict_total)*100
pred_percent = sorted((value, key) for (key,value) in pred_percent.items())
pred_percent.reverse()
print()
print(pred_percent)


#make predictions on the testing set
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('Naive', GaussianNB()))
# evaluate each model in turn for robustness using 10-fold cross validation
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#information for all models used: Precision, accuracy, recall, and error scores
print("\nBased on decision tree classifier")
print("\n")
print ("Precision score is ", metrics.precision_score(y_test,decision_tree_split_predicted, average='weighted')*100)
print ("Accuracy score is ", metrics.accuracy_score(y_test,decision_tree_split_predicted)*100)
print ("Recall score is ", metrics.recall_score(y_test,decision_tree_split_predicted, average='weighted')*100)
print ("Error score is ", (metrics.mean_squared_error(y_test, decision_tree_split_predicted)))


print("\nBased on Naive Bayes' classifier")
print("\n")
print ("Precision score is ", metrics.precision_score(y_test,naive_split_predicted, average='weighted')*100)
print ("Accuracy score is ", metrics.accuracy_score(y_test,naive_split_predicted)*100)
print ("Recall score is ", metrics.recall_score(y_test,naive_split_predicted, average='weighted')*100)
print ("Error score is ", (metrics.mean_squared_error(y_test, naive_split_predicted)))

print("\nBased on KNN classifier")
print("\n")
print("KNN accuracy score")
print ("Precision score is ", metrics.precision_score(y_test,knn_split_predicted, average='weighted')*100)
print ("Accuracy score is ", metrics.accuracy_score(y_test,knn_split_predicted)*100)
print ("Recall score is ", metrics.recall_score(y_test,knn_split_predicted, average='weighted')*100)
print ("Error score is ", (metrics.mean_squared_error(y_test, knn_split_predicted)))