# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step 1 - Start.

step 2 - Data preparation.

step 3 - Hypothesis Definition.

step 4 - Cost Function.

step 5 - Parameter Update Rule.

step 6 - Iterative Training.

step 7 - Model evaluation.

step 8 - End. 

## Program:
/*
Program to implement the multivariate linear regression model for
Developed by: ANUVIND KRISHNA K
RegisterNumber: 212223080004

 

import numpy as np

from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import SGDRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()



x= data.data[:,:3]



y=np.column_stack((data.target,data.data[:,6]))



x_train, x_test, y_train,y_test = train_test_split(x,y, test_size



scaler_x = StandardScaler()

scaler_y = StandardScaler()



x_train = scaler_x.fit_transform(x_train)

x_test = scaler_x.fit_transform(x_test)

y_train = scaler_y.fit_transform(y_train)

y_test = scaler_y.fit_transform(y_test)


sgd = SGDRegressor(max_iter=1000, tol = 1e-3)


multi_output_sgd= MultiOutputRegressor(sgd)


multi_output_sgd.fit(x_train, y_train)


y_pred =multi_output_sgd.predict(x_test)


y_pred = scaler_y.inverse_transform(y_pred)

y_test = scaler_y.inverse_transform(y_test)
print(y_pred)

[[ 1.04860312 35.69231257]

[ 1.49909521 35.72530255]

[ 2.35760015 35.50646978]
...

[ 4.47157887 35.06594388]

[ 1.70991815 35.75406191]



Mean Squared Error: 2.560165984862198

mse = mean_squared_error(y_test,y_pred)

print("Mean Squared Error:",mse)

Mean Squared Error: 2.560165984862198

print("\nPredictions:\n",y_pred[:5])
Predictions:
[[ 1.04860312 35.69231257]
[ 1.49909521 35.72530255]
[ 2.35760015 35.50646978]
[ 2.73967825 35.37568192]
[ 2.10914107 35.63894336]]
*/

## Output:
https://github.com/user-attachments/assets/1461a831-51d7-4d92-96a9-841c8b825fb5

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
