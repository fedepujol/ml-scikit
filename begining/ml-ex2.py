#!/bin/usr/env/python

from sklearn import svm
from sklearn import datasets

#Para hacer pesistente el dataset
import pickle

#Model Persistence
#Para guardar un dataset en scikit, se utiliza la persistencia de python (pickle)

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

#Se hace persistente el dataset iris
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2 = predict(X[0:1])
y[0]