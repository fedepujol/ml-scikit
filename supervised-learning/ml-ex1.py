#!/bin/usr/env/python

#Problema k-NN de vecinos. 
#Principio de la Maldicion de Dimensionalidad ("The Curse of Dimensionality").

from sklear import datasets
import numpy as np

#Para el aprendimiento de los clasificadores, se tienen que dividir los datasets
#en entrenamiento y test. Para que el resultado sea comprobable.

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

np.random.seed(0)
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_train = iris_y[indices[:-10]]
iris_y_test = iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClasifier
knn = KNeighborsClasifier()
knn.fit(iris_x_train, iris_y_train)
knn.predict(iris_x_test)
#array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])

iris_y_test
#array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])

#------------------------------------------------------------------
#------------------------------------------------------------------

#Regresion Lineal
from sklearn import LinearRegression

diabetes = datasets.load_diabetes()
diabetes_x_train = diabetes.data[:-20]
diabetes_y_train = diabetes.target[:-20]
diabetes_x_test = diabetes.data[-20:]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_x_train, diabetes_y_train)

print(regr.coef_)

#Mean Square Error
np.mean((regr.predict(diabetes_x_test) - diabetes_y_test)**2)


#------------------------------------------------------------------
#------------------------------------------------------------------
#Shrinkage

#Si se tiene pocos datos por dimension, el ruido en las mediciones provoca una varianza muy alta




