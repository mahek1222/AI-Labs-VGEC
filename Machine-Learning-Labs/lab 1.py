# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import pandas.core.config_init  # pyright: ignore[reportUnusedImport] # noqa: F401
# from pandas.core.api import (
#           # dtype
#           ArrowDtype,
#           Int8Dtype,
#           Int16Dtype,
#           Int32Dtype,
#           Int64Dtype,
#           UInt8Dtype,
#           UInt16Dtype,
#           UInt32Dtype,
#           UInt64Dtype,
#          Float32Dtype,
#           Float64Dtype,
#          CategoricalDtype,
#           PeriodDtype,)
# # Load the dataset
# df = pd.read_csv(r"C:/Users/DELL/Downloads/appliance_energy.csv",
#                  sep=None,
#                  engine="python",
#                  encoding="latin1",
#                  on_bad_lines="skip")
# # Display the first few rows to understand the structure
# print(df.head(10))
# df.shape

# # Check for missing values
# print(df.isnull().sum()) 
# # Features (independent variable) and target (dependent variable)
# # Independent variable (Temperature)
# X: pd.DataFrame = df[['Temperature (Â°C)']]  
# # Dependent variable (Energy Consumption)
# y = df['Energy Consumption (kWh)']  

# # # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=24)
# print(X_test)

# print(y_test)

# # Create a Linear Regression model
# model = LinearRegression()
# # Train the model
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)
# y_pred

# y_test

# # Calculate Mean Squared Error
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
# # Calculate R-Squared value
# r2 = r2_score(y_test, y_pred)
# print(f"R-Squared: {r2}")

# model.predict([[31.993476]])

# # Plot the test data and regression line
# plt.scatter(X_test, y_test, color='blue', label='Test Data')  # Actual data
# plt.plot(X_test, y_pred, color='red', label='Regression Line')  # Predicted data
# plt.xlabel('Temperature (Â°C)')
# plt.ylabel('Energy Consumption (kWh)')
# plt.legend()
# plt.title('Energy Consumption Prediction using Simple Linear Regression')
# plt.show()

#linear regression for experience salary dataset 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas.core.config_init  # pyright: ignore[reportUnusedImport] # noqa: F401
from pandas.core.api import (
          # dtype
          ArrowDtype,
          Int8Dtype,
          Int16Dtype,
          Int32Dtype,
          Int64Dtype,
          UInt8Dtype,
          UInt16Dtype,
          UInt32Dtype,
          UInt64Dtype,
         Float32Dtype,
          Float64Dtype,
         CategoricalDtype,
          PeriodDtype,)
# Load the dataset
df = pd.read_csv(r"C:/Users/DELL/Downloads/experience_salary_100_rows.csv",
                 sep=None,
                 engine="python",
                 encoding="latin1",
                 on_bad_lines="skip")
# Display the first few rows to understand the structure
print(df.head(10))
df.shape

# Check for missing values
print(df.isnull().sum()) 
# Features (independent variable) and target (dependent variable)
# Independent variable (Temperature)
X: pd.DataFrame = df[['Experience Years']]  
# Dependent variable (Energy Consumption)
y = df['Salary']  

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=24)
print(X_test)

print(y_test)

# Create a Linear Regression model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred

y_test

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Calculate R-Squared value
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")

model.predict([[31.993476]])

# Plot the test data and regression line
plt.scatter(X_test, y_test, color='blue', label='Test Data')  # Actual data
plt.plot(X_test, y_pred, color='red', label='Regression Line')  # Predicted data
plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.legend()
plt.title('salary Prediction using Simple Linear Regression')
plt.show()
