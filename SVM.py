
import keras 
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import sgd
from joblib import dump, load

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import itertools


from collections import Counter

from sklearn.metrics import confusion_matrix


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics



(x_train,y_train),(x_test,y_test)=mnist.load_data()



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)



#y_train = to_categorical(y_train, num_classes=10)
#y_test = to_categorical(y_test, num_classes=10)


C=1

model1=svm.SVC(kernel='linear', C=C, max_iter=20)
model2=svm.SVC(kernel='rbf', gamma=0.7, C=C)
model3=svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)


#model1.fit(x_train,y_train)

#dump(model1,'./SVM_models/linear.joblib')

#model1=load('./SVM_models/linear.joblib')


#predicted = model1.predict(x_test)

#report=metrics.classification_report(y_test, predicted)

pre=[]
recall=[]	






#print("Classification report for classifier %s:\n%s\n" % (model1,report))

#disp = metrics.plot_confusion_matrix(model1, x_test, y_test)
#disp.figure_.suptitle("Confusion Matrix")

#print("Confusion matrix:\n%s" % disp.confusion_matrix)

#print(report[0])



for C in range(1,10):
	model=svm.SVC(kernel='poly', degree=3, gamma='auto', C=(C*10+(C==0)*1),max_iter=500)
	print("Entrenando para C: ",C*10+(C==0)*1)

	model.fit(x_train,y_train)
	predicted = model.predict(x_test)

	report=metrics.classification_report(y_test, predicted)
	print("Classification report for classifier %s:\n%s\n" % (model,report))

	disp = metrics.plot_confusion_matrix(model2, x_test, y_test)
	disp.figure_.suptitle("Confusion Matrix")

	plt.show()
	
	dump(model1,'./SVM_models/poly.joblib')







