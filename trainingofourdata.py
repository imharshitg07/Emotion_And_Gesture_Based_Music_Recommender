import numpy as np
import cv2
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
# we have imported to_categorical to make our one hot vectors
from keras.layers import Dense,Input
from keras.models import Model
import os

initialized = False
initial_size = -1


resultingLabels = []
#resultingLabels
dictionary ={}


cnt = 0
#  this cnt will map every word in dixtionary to a unique integer associated with it



#cnt

for i in os.listdir():
	# print(i)
	# the os.listdir lists all the files in the current directory or folder
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
		if not(initialized):
			initialized = True
			X = np.load(i)
			size = X.shape[0]
			#  size of the x 
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
			# y is the name or label associated with the data anad make the lisut of same size as x

		else:
			X = np.concatenate((X,np.load(i)))
			y = np.concatenate((y,np.array([i.split('.')[0]]*size).reshape(-1,1)))

		resultingLabels.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = cnt
		cnt += 1

for i in range(y.shape[0]):
	y[i,0] = dictionary[y[i,0]]

y = np.array(y,dtype='int32')



y = to_categorical(y)


NEWX = X.copy()
NEWY = y.copy()


countingVariable = 0



c = np.arange(X.shape[0])
np.random.shuffle(c)




for i in c:
	NEWX[countingVariable] = X[i]
	NEWY[countingVariable] = y[i]

	countingVariable += 1

name = Input(shape=(X.shape[1]))


m = Dense(512,activation='relu')(name)
m = Dense(256,activation = 'relu')(m)


output = Dense(y.shape[1],activation='softmax')(m)

model = Model(inputs = name, outputs = output)


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

model.fit(X,y,epochs = 40)


model.save("model.h5")
np.save("labels.npy",np.array(resultingLabels))