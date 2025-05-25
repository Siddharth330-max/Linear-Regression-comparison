import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression   
 

## Step 1 -: Load the dataset  
data = pd.read_csv('Weight-Height for linear regression.csv')
X = data['Weight'].values.reshape(-1,1) 
y = data['Height'].values  

## Step 2 -: Performing train-test split and Standardisation
X_train , X_test , y_train, y_test = train_test_split(X , y, test_size=0.20 ,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  

## Step 3 -: Model Training  
model = LinearRegression()            ##This Linear Regression is from sklearn library  
model.fit(X_train_scaled , y_train)  


##Step 4 -: Finding predictions
y_pred_train = model.predict(X_train_scaled)  
y_pred_test = model.predict(X_test_scaled)   

##Step 5 -: Performance metrics  
def mse(true_values , predicted_values):
    score = np.mean((true_values - predicted_values)**2)
    return score  

print(f"Mean Squared Error for training data :")
print(mse(y_train , y_pred_train))  

print(f"Mean Squared Error for test data : ")  
print(mse(y_test , y_pred_test))   

##Step 6 -: Visualization  
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')

##Sorting the X values for smooth line
 
X_all = np.concatenate([X_train, X_test])
X_all_sorted = np.sort(X_all, axis=0)

X_all_scaled = scaler.transform(X_all_sorted)

plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_all_sorted, model.predict(X_all_scaled), color='red', label='Best Fit Line')

plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Linear Regression - Height vs Weight')
plt.legend()
plt.grid(True)
plt.show()

