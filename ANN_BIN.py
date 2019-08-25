
# coding: utf-8

# In[77]:

import numpy as np
import os
import csv
import sys
import shutil
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import hashlib
from sklearn.datasets.base import Bunch




# In[78]:

def load_data(module_path, data_file_name):
    with open(join(module_path, '/home/maanas/Documents/project/csv', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names


# In[79]:

def load_iris(return_X_y=False):
    module_path = dirname('eeg.csv')
    data, target, target_names = load_data(module_path, 'eeg.csv')
    iris_csv_filename = join(module_path, '/home/maanas/Documents/project/csv', 'eeg.csv')
    print (iris_csv_filename)

    with open(join(module_path, '/home/maanas/Documents/project/csv', 'iris.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['1', '2',
                                '3', '4','5','6','7','8','9','10','11'],filename=iris_csv_filename)


# In[80]:

#load the iris dataset
iris = load_iris()
print(iris)
#our inputs will contain 4 features
X = iris.data[:, 0:11]
#the labels are the following
y = iris.target
#print the distinct y labels
print(X)
print(y)
print(np.unique(y))
print (iris)
print(y)


# In[82]:

#One Hot Encode our Y:
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
Y = encoder.fit_transform(y)
print (Y)


# In[83]:

from keras.models import Sequential #Sequential Models
from keras.layers import Dense #Dense Fully Connected Layer Type
from keras.optimizers import SGD #Stochastic Gradient Descent Optimizer


# In[84]:

def create_network():
    model = Sequential()
    model.add(Dense(5, input_shape=(11,), activation='relu'))
    model.add(Dense(10, activation='softmax'))
        
    #stochastic gradient descent
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
neural_network = create_network()
neural_network.fit(X,Y, epochs=1000, batch_size=100)


# In[85]:

import numpy as np
np.set_printoptions(suppress=True)
print (Y[0:400])
neural_network.predict(X[0:400], batch_size=32, verbose=0)

