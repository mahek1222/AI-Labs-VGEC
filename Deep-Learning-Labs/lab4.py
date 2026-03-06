import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , SimpleRNN, Dropout
from sklearn.model_selection import train_test_split

data=pd.read_csv(r"C:\Users\DELL\Downloads\waste management.csv")

data.head()

sea_level_rise=data['sea_level_rise'].values

scaler= MinMaxScaler(feature_range=(0,1))
sea_level_rise_scaled= scaler.fit_transform(sea_level_rise.reshape(-1,1))

def create_sequence(data,sequence_length):
    sequence=[]
    labels=[]
    for i in range(len(data)-sequence_length):
        sequence.append(data[i:i + sequence_length])
        labels.append(data[i+sequence_length])
    return np.array(sequences), np.array(labels)

sequence_length=10
x,y=create_sequence(sea_level_rise,sequence_length)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(f"TRaining shape : {x_train.shape},testing shape :{x_test.shape}")

model = Sequential([
    SimpleRNN(50,activation='tanh',return_sequence=false,input_shape=(sequence_length,1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam',loss='mean_square_error',metrics=['mae'])
model.summary()

history= model.fit(
    x_train,y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test,y_test),
    verbose=1
)

loss,mae=model.evaluate(x_test,y_test,verbose=0)
print(f"Test Loss: {loss:.4f},Test MAE:{mae:.4f}")

predictions =model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_original=scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure(figsize=(10,6))
plt.plot(y_test_original,label="Actual values")
plt.plot(predictions,label="prediction values",linestyle='--')
plt.title("waste volume prediction")
plt.xlabel("time")
plt.ylabel("sea_level_rise")
plt.legend()
plt.show()