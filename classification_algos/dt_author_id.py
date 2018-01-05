#!/usr/bin/python

"""
    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### Decision Tree Classifier training and testing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

clf = tree.DecisionTreeClassifier(min_samples_split = 40)
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

acc = accuracy_score(labels_test,pred)
print(acc)  ###0.979

matrix = confusion_matrix(labels_test,pred)
print(matrix)

report = classification_report(labels_test,pred)
print(report)

#########################################################
