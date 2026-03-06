# create dataset using numpy and performs opertaion on it 
import numpy as np
import pandas as pd 

house_price_prediction = np.array([2000000,40000000,5000000,2500000,3000000,65000000])
print("predection of house price: ")
print(house_price_prediction)
total = np.sum(house_price_prediction)
print("total price of predected house price")
print(total)
mean= np.mean(house_price_prediction)
print("mean of price ")
print(mean)
median= np.median(house_price_prediction)
print("median of price :")
print(median)
standard_deviation= np.std(house_price_prediction)
print("standared deviation :")
print(standard_deviation)
reshape = house_price_prediction.reshape((3,2))
print("Here reshape of series data ")
print(reshape)