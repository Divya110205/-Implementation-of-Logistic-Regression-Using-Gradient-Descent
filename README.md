# EX 5-Implementation-of-Logistic-Regression-Using-Gradient-Descent
## DATE:05.10.23
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
![1](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/798e7b65-ad53-46b6-b4c8-0315e18db1b6)

### Array Value Of Y:
![2](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/0fa455ab-0634-4ee4-a11c-a6d1c981000c)

### Exam-1 Score Graph:
![3](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/98c209b6-901b-492f-8869-13337099b1a8)

### Sigmoid Function Graph:
![4](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/3135d387-2f55-4588-8f08-26c53da327c8)

### X_train_grad Value:
![5](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/c4d2daeb-3259-4f40-8884-413ff5d678de)

### Y_train_grad Value:
![6](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/075bb5cb-394c-45b9-8aa8-9845e045133f)

### Print res.x:
![7](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/bce7850d-0921-4947-9814-207a10e7709d)

### Desicion Boundary-Graph For Exam Score:
![8](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/78d98d8c-6c27-4953-81e0-bda1a5f6bd76)

### Probability Value:
![9](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/c359a6ee-d919-47ff-8823-f5e053a06947)

### Prediction Value Of Mean:
![10](https://github.com/Divya110205/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404855/945abf34-7f20-4f89-a60f-f747a45eedd2)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

