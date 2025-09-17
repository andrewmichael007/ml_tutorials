# importing various libraries
import numpy as np
import matplotlib.pyplot as plt

# Data: shoe size, weight (features) and height (target(y-hat))
# x is an array containing the inputs that is the shoe size and weight
# each row is a training example and each column is a feature
X = np.array([
    [7, 60],  # (shoe size, weight in kg)
    [8, 65],
    [9, 70],
    [10, 75],
    [11, 80]
])

#test and debug
#first training example
print( f"first training example: ", (X[0]) )  #👉 [ 7, 60 ]

#second training example
print( f"second training example: ", (X[1]))  #👉 [ 8, 65 ]

#third training example
print( f"third training example : ", (X[2]))  #👉 [ 9, 70 ]

#first feature of the first training example ( shoe size )
print( f"first feature of the first training example: ", ((X[0, 0])) ) #👉 7

#second feature of the second training example ( weight)
print( f"second feature of the second training example: ", (X[0, 1]) ) #👉 60

#first feature of the second training example 
print( f"first feature of the second training example: ", (X[1, 0]) ) #👉 8

#second feature of the second training example
print( f"second feature of the second training example: ", (X[1, 1]) ) #👉 65

y = np.array([160, 165, 170, 175, 180])   #Heights (cm)

m, n = X.shape  # m = number training set, n = number of features

# Initialize weights and bias
W = np.array([1.0, 1.0])# Random values for w1, w2
b = 0.0

# Hyperparameters
learning_rate = 0.0001  # Learning rate (step size)
number_of_iterations = 1000  # Number of iterations

# Store cost history
cost_history = []

# Gradient Descent Algorithm
for each in range(number_of_iterations):
    # Compute predictions: y_hat = X * W + b with numpy dot function
    y_hat = np.dot(X, W) + b

    # Compute error
    errors = y_hat - y # Sum of errors (not needed for vectorized form)

    # Compute cost function (Mean Squared Error)
    cost_function = (1 / (2 * m)) * np.sum(errors ** 2)
    cost_history.append(cost_function)
    
    # Compute gradients
    dW = (1 / m) * np.dot(X.T, errors)  # Partial derivatives w.r.t W
    db = (1 / m) * np.sum(errors)  # Partial derivative w.r.t b

    # Update parameters
    W -= learning_rate * dW
    b -= learning_rate * db

    # Print cost every 100 iterations
    if each % 100 == 0:
        print(f"Iteration {each}: Cost = {cost_function:.4f}, W = {W}, b = {b:.4f}")

# Final parameters after training
print("\nFinal Weights:", W)
print("Final Bias:", b)

# Plot cost function convergence
plt.plot(range(number_of_iterations), cost_history, '-b', linewidth=2)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence for Multivariate Linear Regression")
plt.show()





