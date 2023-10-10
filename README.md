# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. import numpy as np.
3. Give the header to the data.
4. Find the profit of population.
5. Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
6. End the program.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: R.THIVAKAR
RegisterNumber: 212222240109
```
```
python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('dataset/ex1.txt', header = None)

plt.scatter(data[0], data[1])
plt.xticks(np.arange(5, 30, step = 5))
plt.yticks(np.arange(-5, 30, step = 5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x, y, theta):
    """
    Test in a numpy array x, y theta and generate the cost function
    in a linear regression model
    """
    m = len(y) # length of the training data
    h = X.dot(theta) # hypothesis
    square_err = (h - y) ** 2
    
    return 1/(2*m) * np.sum(square_err) #returning
    
data_n = data.values
m = data_n[:, 0].size
X = np.append(np.ones((m, 1)), data_n[:, 0].reshape(m, 1), axis = 1)
y = data_n[:, 1].reshape(m, 1)
theta = np.zeros((2, 1))

computeCost(X, y, theta) # Call the function

def gradientDescent(x, y, theta, alpha, num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_oters gradient steps
    with learning rate of alpha
    
    return theta and the list of the cost of theta during each iteration
    """
    
    m = len(y)
    J_history = []
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * 1/m * error
        theta -= descent
        J_history.append(computeCost(X, y, theta))
        
    return theta, J_history
    
theta, J_history = gradientDescent(X, y, theta, 0.01, 1500)
print("h(x) ="+str(round(theta[0, 0], 2))+" + "+str(round(theta[1, 0], 2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0], data[1])
x_value = [x for x in range(25)]
y_value = [y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value, y_value, color = "r")
plt.xticks(np.arange(5, 30, step = 5))
plt.yticks(np.arange(-5, 30, step = 5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x, theta):
    """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
    """
    
    predictions = np.dot(theta.transpose(), x)
    
    return predictions[0]
    
predict1 = predict(np.array([1, 3.5]), theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1, 0)))

predict2 = predict(np.array([1, 7]), theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2, 0)))
```
## Output:

#### Profit Prediction Graph:

![Screenshot from 2023-10-05 10-18-12](https://github.com/Gchethankumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118348224/be61d168-0640-4ed5-bac7-59ae16c705dc)


#### Compute cost value:

![Screenshot from 2023-10-05 10-18-45](https://github.com/Gchethankumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118348224/3f0d3b66-8556-4056-952f-955e30abccc2)


#### h(x) value:

![Screenshot from 2023-10-05 10-18-52](https://github.com/Gchethankumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118348224/484e0894-59d8-43bd-84e1-533425851a8f)


#### Cost function using gradient descent Graph:

![Screenshot from 2023-10-05 10-19-27](https://github.com/Gchethankumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118348224/4e50a4c3-9022-45b3-b17c-97e242c55932)


#### Profit prediction Graph:

![Screenshot from 2023-10-05 10-19-36](https://github.com/Gchethankumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118348224/f3484d4b-3e87-4b36-9237-83b00b80a646)


#### Profit for the Population 35,000 :

![Screenshot from 2023-10-05 10-19-47](https://github.com/Gchethankumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118348224/ea7e4ec8-e998-4678-b592-8b1fdd16b006)


#### Profit for the Population 70,000 :

![Screenshot from 2023-10-05 10-19-53](https://github.com/Gchethankumar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118348224/d4f0c3d6-b088-4c1e-b44d-5e9625554659)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
