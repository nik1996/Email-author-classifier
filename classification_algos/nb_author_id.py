#!/usr/bin/python

"""
   Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
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
### Naive Bayes Classifier training and testing ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print("training time: %s", round(time()-t0, 3))

t0=time()
pred = clf.predict(features_test)
print("prediction time: %s", round(time()-t0, 3))

acc = accuracy_score(labels_test, pred)

print(acc)  ###0.9732

matrix = confusion_matrix(labels_test,pred)
print(matrix)

report = classification_report(labels_test,pred)
print(report)

#########################################################
