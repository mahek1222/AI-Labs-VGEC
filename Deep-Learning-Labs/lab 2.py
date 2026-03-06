import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

data= pd.read_csv(r"C:\Users\DELL\Downloads\climate_data.csv")
x = data.drop(columns=["temperature"])
y =data["temperature"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model =Sequential()
model.add(Dense(64 ,input_dim=x_train.shape[1],activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(optimizer ='adam',loss='mse',metrics=['mae'])

early_stopping =EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

history = model.fit(
    x_train,y_train,
    validation_split=0.2,
    epochs=50,
    batch_size= 32,
    callbacks=[early_stopping]
)

test_loss,test_mae=model.evaluate(x_test,y_test)
print(f"Test loss :{test_loss},Test MAE :{test_mae}")

prediction=model.predict(x_test)
print("preadected :",prediction[:5])
print("Actual Values :",y_test[:5].values)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='tain loss')

plt.plot(history.history['val_loss'],label='validatio loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.title('training validation loss')
plt.show()