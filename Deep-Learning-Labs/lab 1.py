from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
df = pd.read_csv(r"C:\Users\DELL\Downloads\solar_power_output.csv")
x= df.drop('crop_yield',axis=1)
y = df['copy_yield']

scaler =MinMaxScaler()
x_scaled=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

model =Sequential()
model.add (Dense(64, input_dim=x_train.shape[1],activation="relu"))
model.add (Dense(32, activation='relu'))
model.add (Dense(1,activation='linear'))

model.compile(optimizer='adam',loss ='mean square error ')

model.fit (x_train,y_train,epochs=100,batch_size= 10)