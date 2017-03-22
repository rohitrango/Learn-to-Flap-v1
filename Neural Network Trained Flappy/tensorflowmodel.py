import tensorflow as tf
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers
from keras import backend
import os.path
from keras.models import load_model
batch_size = 1000
Data = np.loadtxt('train_new.csv',delimiter = ',')
train_x = list(map(lambda x:x[0:3],Data))
train_x = np.matrix(train_x)
train_y = list(map(lambda x:x[3:5],Data))
train_y = np.matrix(train_y)
# train_y = train_y.transpose()
if os.path.isfile('my_model.h5'):
	model = load_model('my_model.h5')
	print('hello')
else:
	model = Sequential()
	model.add(Dense(units=20,kernel_initializer=initializers.random_normal(stddev=0.01),input_dim=3))
	model.add(Activation('relu'))
	model.add(Dense(units=30, kernel_initializer=initializers.random_normal(stddev=0.01)))
	model.add(Activation('relu'))
	model.add(Dense(units=20, kernel_initializer=initializers.random_normal(stddev=0.01)))
	model.add(Activation('relu'))
	# model.add(Dense(units=10))
	# model.add(Activation('softmax'))
	model.add(Dense(units=2,kernel_initializer=initializers.random_normal(stddev=0.01)))
	model.compile(loss='mse',
	              optimizer='adam')
print('Model Ready')

model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=100, verbose=1)
model.save('my_model.h5') 