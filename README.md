# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DIVYA.A
RegisterNumber:  212222230034
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt",delimiter=",")
X=data[:,[0,1]]
y=data[:,2]

# Array Value Of X:
X[:5]

# Array Value Of Y:
y[:5]

# Exam-1 Score Graph:
plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1],label="Admitted")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

# Sigmoid Function Graph:
def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

# X_train_grad Value:
def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

# Y_train_grad Value:
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

# Print res.x:
def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

# Desicion Boundary-Graph For Exam Score:
def plotDecisionBoundary(theta,X,y):
  x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max = X[:,0].min()-1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y == 1][:,0],X[y == 1][:,1],label='Admitted')
  plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label='Not admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

# Probability Value:
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return(prob>=0.5).astype(int)

# Prediction Value Of Mean:
np.mean(predict(res.x,X)==y)


```

## Output:
### Array Value Of X:
![1](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/68c08599-52cd-4526-a92c-f9fd5d5e0c3c)

### Array Value Of Y:
![2](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/7b0863c7-6436-4cc3-a1f6-c74e7740e343)

### Exam-1 Score Graph:
![3](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/00a6ad23-39f5-4dd4-b20d-40ba26564be9)

### Sigmoid Function Graph:
![4](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/da8baadd-043b-4705-a338-cc75c673ac17)

### X_train_grad Value:
![5](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/250cccd4-9431-4016-a46e-bde82b587a44)

### Y_train_grad Value:
![6](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/605c6746-ca6b-4e59-aaee-eb7f97199d0f)

### Print res.x:
![7](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/9b8451c3-54f8-4982-9b45-4c34eb877216)

### Desicion Boundary-Graph For Exam Score:
![8](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/a2166ed1-fa04-4e39-8d9f-773b21ffd1a4)

### Probability Value:
![9](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/3e8e6842-1504-4ace-9f29-52a0729d5f66)

### Prediction Value Of Mean:
![10](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/efd78fb0-2230-4bc1-8fe9-f172a6f3e625)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

