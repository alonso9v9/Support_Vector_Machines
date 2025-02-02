# -*- coding: utf-8 -*-
"""Keras.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GpNWa5h1mJMiJmm6Wl6dDVSCk7B7CYQT
"""



import keras 
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import sgd


import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import itertools


from collections import Counter

from sklearn.metrics import confusion_matrix


#######################################################

plt.grid(False)



(x_train,y_train),(x_test,y_test)=mnist.load_data()



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(60000, 28,28)
x_test = x_test.reshape(10000, 28,28)

print(y_test[0])
print(y_train[0])

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(y_test[0])
print(y_train[0])


#Definición del modelo
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.summary()


#Aprendizaje, entrenamiento y evaluación

batch_size = 20
num_classes = 10
epochs=15
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.00009),
              metrics=['accuracy'])



history = model.fit(x_train, y_train,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test),
                    batch_size=batch_size)



loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Precisión Entrenamiento: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Precisión Prueba:  {:.4f}".format(accuracy))


model.save('numbers.h5')

plt.style.use('ggplot') # 

def plot_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5)) 

    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Entrenamiento prec')
    plt.plot(x, val_acc, 'r', label='Validacion prec')
    plt.title('Precision Entrenamiento y validacion')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Perdida Entrenamiento')
    plt.plot(x, val_loss, 'r', label='Perdida Validacion')
    plt.title('Perdida Entrenamiento y validacion')					
    plt.legend()

    

    plt.show()
    

plot_history(history)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

#predicciones
#imagen para poder comprobar predicción dado el modelo hecho
predictions = model.predict(x_test)
#plt.imshow(x_test[11], cmap=plt.cm.binary)
print(y_test[11])
#realiza la predicción
np.argmax(predictions[11])

# ver la matriz de confusión 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta matriz imprime y plotea la matriz de confusión.
    Normalización puede ser m aplicada al cambiar `normalize=True`.
    """
    
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Observación')
    plt.xlabel('Predicción')
    plt.show()
    


# predice los valores de los datos de validación
Y_pred = model.predict(x_test)
# Convierta las clases de predicciones en vectores 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
#Convertir observaciones de validación a vectores 
Y_true = np.argmax(y_test, axis = 1) 
# calcula la matriz de confusión
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plotea la matriz de confusión
plot_confusion_matrix(confusion_mtx, classes = range(10))


