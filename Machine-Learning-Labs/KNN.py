import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"C:/Users/DELL/Downloads/diabetes.csv")
# Display the first few rows
print(data.head())
# Check for missing values
print(data.isnull().sum())
# Handle missing values if necessary
data.fillna(data.mean(), inplace=True)

data.shape

# Define features and target variable
X = data[['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI']]
y = data['Outcome']  # 1 for effective reduction, 0 otherwise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

a = -1
k = 0
for i in range(1,10):
    m = KNeighborsClassifier(n_neighbors=i)
    m.fit(X_train,y_train)
    s = m.score(X_test,y_test)
    if s>a:
        a = s
        k = i

        # Set number of neighbors
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred 

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Effective', 'Effective'], 
            yticklabels=['Not Effective', 'Effective'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Not Effective', 'Effective']))

# Trying different values for k
accuracy_scores = []
k_values = range(1, 20)

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs. k
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors')
plt.show()