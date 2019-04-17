#!/bin/usr/env/python

#Importo los datasets de ejemplos
from sklearn import dataset

#svm para usar en la prediccion
from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()

#Muestra los datos que se pueden usar para la clasificacion (matriz)
print(digits.data)

#Muesta la imagen correspondiente a cada digito (da un array)
digits.target

#Como los datos se muestran en forma 2D (n_samples, n_features)
#los datos originales pueden tener otra forma. Ej: (8, 8). (da un array)
digits.images[0]

#En python, un estimador de clasificacion, es un objeto que implementa los metodos
# fit(X, y) y predict(T).
# El objecto SVC (Support Vector Classification). Su contructor toma como argumentos
# los parametros del modelo.
clf = svm.SVC(gamma=0.001, C=100.)

#Clf (classifier) tiene que aprender con el modelo. Entonces se le pasa al metodo fit()
#los datos y la imagenes del dataset. Se le pasan todos excepto por uno que es el que se
#usa para la prediccion.
clf.fit(digits.data[:-1], digits.target[:-1])

#Para la prediccion, se usa la ultima imagen de digits.data. 
#Y se determina, desde el set de entrenamiento, la imagen que mejor
#represente la ultima imagen de digits.data
clf.preditct(digits.data[-1:])