#!/bin/usr/env/python

#Type Casting
#Regression targets se transforman a float64 y los de clasificacion se mantienen con el formato
import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 200)
X = np.array(X, dtype='float32')

#Muestra el tipo de datos Float32.
#Si no se acalara, se transforma en Float64
X.dtype

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)

#Al no ser especificado el tipo float32, se imprime
#float64
X_new.dtype