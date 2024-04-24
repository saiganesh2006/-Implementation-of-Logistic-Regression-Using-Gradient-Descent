# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: D.B.V. SAI GANESH
RegisterNumber:  212223240025
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset= dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y ):
    h = sigmoid(X.dot(theta)) 
    return -np.sum(y *np.log(h)+ (1- y) *np.log(1-h))
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot (h-y) /m
        theta-=alpha * gradient
    return theta
theta= gradient_descent (theta,X,y,alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where( h >= 0.5,1 , 0)
    return y_pred

y_pred= predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y) 
print("Accuracy:", accuracy)
print(Y)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
### DATASET:
![image](https://github.com/saiganesh2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742342/aff65980-3f28-43fb-87f2-3b9c04bcc864)

### Labelling data:
![image](https://github.com/saiganesh2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742342/a0594701-4491-4dc4-abd5-27db857b44a1)

### Lablling the column:
![image](https://github.com/saiganesh2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742342/e2fc4c59-0930-4f75-8e01-ea596614a294)

### Dependent Variables:
![image](https://github.com/saiganesh2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742342/d3a47afa-ad0d-4243-a0b7-fbcd3e815f15)

### Accuracy:
![image](https://github.com/saiganesh2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742342/ee265fd4-158a-4a61-8b73-ed5d39945164)

### Y:
![image](https://github.com/saiganesh2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742342/844eaea5-0f1e-4c0d-a3b5-970810c16b6c)

### Y_pred:
![image](https://github.com/saiganesh2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742342/3bdb6e06-66cd-4337-8e7d-b6d65f3994d4)

### New Predicted data:
![image](https://github.com/saiganesh2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742342/39580a6d-b292-4ea8-a792-c2fc47046690)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

