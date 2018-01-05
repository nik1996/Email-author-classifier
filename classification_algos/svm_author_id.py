#!/usr/bin/python

"""
    Use a SVM to identify emails from the Enron corpus by their authors:
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
### SVM training and testing ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

clf = SVC(kernel = 'rbf',C = 10000.0)
t0 = time()
clf = clf.fit(features_train,labels_train)
print("training time: ", round(time()-t0, 3))

t0 = time()
pred = clf.predict(features_test)
print("prediction time: ", round(time()-t0, 3))

acc = accuracy_score(labels_test,pred)  

print(acc)   ###0.991

print(acc)
#########################################################
