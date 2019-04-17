#!/bin/usr/env/python

#Si se usan multiclass, el resultado y el aprendizaje dependen del formato de los datos de entrada.

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(gamma='scale', random_state=0))
#...

#Al metodo fit del estimador se le provee un array unidimensional multiclase, 
#el metodo predict devuelve la prediccion correpondiente (arrray unidimensional multiclase)

classif.fit(X, y).predict(X)
#array([0, 0, 1, 1, 2])

#Se le pasa un array bidimensional al metodo fit() y se devuelve con el predict() una matriz
y = LabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)
#array([[1, 0, 0],
#      [1, 0, 0],
#      [0, 1, 0],
#      [0, 0, 0],
#      [0, 0, 0]])