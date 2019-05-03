# Crime-Predicting-Tool

Full dataset (I used) can be found here: https://www.kaggle.com/ankkur13/boston-crime-data

A Python application that aggregates and gathers various information from an Excel (csv) file.
Excel file contains code group of criminal activity, year and a SPECIFIC month.
The tool analyzes and create a set of predictions based on data gathered; printing various information about the predicted data. 
To provide further analytics on model robustness, various accuracy tests are also applied. 

Prediction of data is kept in a dictionary. 
Key is the code group for the criminal activity, Value of the key is how often that code appears in the overall prediction dataset.
This is called the prediction data.

There is also a validated dataset where we take a csv file that contains present data.
Python file, count_validated_data, puts each entry of this dataset into a dictionary just like prediction data where:
Key is the code group for the criminal activity, Value of the key is how often that code appears in the overall prediction dataset.
This is called validated data, since it is present data.

Example:
Prediction data comes from information from a csv file that contains data from October 2015-2017.
Validated data comes from information from a csv file that contains data from October 2018.
We take these 2 datasets and compare them to each other to see how accurate our prediction data set is from actual validated data.

Models used: Naive Bayes, K-Nearest Neighbors (KNN), Decision Tree

To demonstrate model scalability,
an output prints out each model's run time.

To demonstrate model robustness,
each model is ran through a 10-fold cross validation, where scoring data is gathered for each fold.

Scoring information for all models used: Precision, accuracy, recall, k-fold cross validation and error scores

-----------------------------------------------------
Run in a shell terminal window

python3 projCS301.py {file_name}

{file_name} - Argument 1 - CSV file used to train the model

Ex: 'crime_06_2015-2017.csv' - Remove quotations
Format to run:
python3 projCS301.py crime_06_2015-2017.csv
