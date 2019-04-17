#!/bin/usr/env/python

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()

clf = SVC(gamma='scale')

#Devuelve un array de enteros, ya que se usa
#iris.target (array de enteros)
clf.fit(iris.data, iris.target)
list(clf.predict(iris.data[:3]))

#Devuelve un array de caracteres, ya que se usa 
#iris.target_names para el fit
clf.fit(iris.data, iris.target_names[iris.target])
list(clf.predict(iris.data[:3]))