#!/bin/usr/env/python

#Actualiacion de parametros
#Los hiper-parametros de un estimador (clf) pueden ser cambiados
#despues de haber sido construido con el metodo set_params().
#Llamar al metodo fit() mas de una vez, borra lo ya aprendido por un fit() anterior.

import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)

X = rng.rand(100, 10)
y = rng.rand(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = SVC()

#Se cambia el kernel a linear cuando ya se construye el estimador
clf.set_params(kernel='linear').fit(X, y)

#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
# kernel='linear', max_iter=-1, probability=False, random_state=None,
# shrinking=True, tol=0.001, verbose=False)

clf.predict(X_test)
#array([1, 0, 1, 1, 0])

#Se lo vuelve a cambiar a rbf y hacer una segunda prediccion
clf.set_params(kernel='rbf', gamma='scale').fit(X, y)
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
# max_iter=-1, probability=False, random_state=None, shrinking=True,
# tol=0.001, verbose=False)

clf.predict(X_test)
#array([0, 0, 0, 1, 0])