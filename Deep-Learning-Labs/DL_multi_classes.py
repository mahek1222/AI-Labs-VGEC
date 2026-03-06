import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras
import tensorflow as tf
from keras.layers import Dense,Flatten
from keras import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train.shape

X_train[0]

y_train[0]

X_train = X_train/255
X_test = X_test/255

X_train[0]

model=Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=20,batch_size=100,validation_split=0.2)

y_prob= model.predict(X_test)

y_prob

y_pred = y_prob.argmax(axis=1)

y_pred

accuracy_score(y_test,y_pred)