import tensorflow as tf
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers
from keras import backend
import os.path
from keras.models import load_model
model = load_model('my_model.h5')
Data = np.loadtxt('train_new.csv',delimiter = ',')
train_x = list(map(lambda x:x[0:3],Data))
train_x = np.matrix(train_x)
train_y = list(map(lambda x:x[3:5],Data))
train_y = np.matrix(train_y)
i = 0
count = 0
for x in train_x:
	y = model.predict(x)
	y1 = np.argmax(y)
	y2 = np.argmax(train_y[i])
	if(y1 == y2):
		count+=1
	i+=1
print(count)

