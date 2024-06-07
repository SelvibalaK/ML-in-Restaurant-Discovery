# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:55:33 2024

@author: Selvibala
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# step 1 => loading the dataset
data = pd.read_csv(r'C:\Users\Selvibala\Downloads\Dataset Cognifyz.csv')

# step 2 => identifing the missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# step 3 => droping the rows with missing values
data.dropna(inplace=True)

# step 4 => droping the irrelevant columns
data.drop(columns=['Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Rating color', 'Rating text', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu'], inplace=True)

# step 5 =>handling the categorical columns
encoder = LabelEncoder()
data['City'] = encoder.fit_transform(data['City'])

# step 6 => assiging the target column
target_column_name = 'Cuisines'
y = data[target_column_name]

# step 7 => droping the target column from the data
X = data.drop(columns=[target_column_name], axis=1)

# step 8 => spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# step 9 => training with Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# step 10 => evaluating the model's performance
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))

# step 11 => analyzing the model's performance across different cuisines
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
