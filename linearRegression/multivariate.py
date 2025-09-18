# importing various libraries
import numpy as np
import matplotlib.pyplot as plt

# Data: shoe size, weight (features) and height (target(y-hat))
# x is an array containing the inputs that is the shoe size and weight
# each row is a training example and each column is a feature
X = np.array([
    [6.2, 58.3],
    [8.1, 67.2],
    [7.5, 62.8],
    [9.3, 74.1],
    [10.2, 78.6],
    [6.8, 60.5],
    [8.7, 69.3],
    [7.1, 59.7],
    [9.8, 76.2],
    [11.2, 82.4],
    [5.9, 55.1],
    [8.4, 68.9],
    [7.8, 64.3],
    [10.6, 79.8],
    [6.5, 57.9],
    [9.1, 72.6],
    [8.9, 70.4],
    [7.3, 61.2],
    [10.9, 81.3],
    [6.1, 56.8],
    [8.6, 67.8],
    [9.5, 75.1],
    [7.9, 63.5],
    [11.4, 83.7],
    [6.7, 59.3],
    [8.2, 66.4],
    [10.1, 77.9],
    [7.6, 62.1],
    [9.7, 74.8],
    [8.8, 69.6],
    [6.4, 58.7],
    [10.8, 80.5],
    [7.2, 60.9],
    [9.2, 73.3],
    [8.5, 68.1],
    [11.1, 82.0],
    [6.9, 61.6],
    [8.3, 67.5],
    [10.4, 78.2],
    [7.7, 63.8],
    [9.6, 75.4],
    [6.3, 57.2],
    [8.0, 65.7],
    [10.7, 79.1],
    [7.4, 62.6],
    [9.4, 74.5],
    [8.1, 66.9],
    [11.3, 83.1],
    [6.6, 58.4],
    [9.9, 76.8],
    [7.0, 60.2],
    [8.9, 70.1],
    [10.5, 78.9],
    [6.8, 61.3],
    [9.3, 73.7],
    [8.7, 68.5],
    [7.5, 63.2],
    [11.0, 81.6],
    [6.2, 56.9],
    [10.2, 77.4],
    [8.4, 67.1],
    [9.1, 72.9],
    [7.8, 64.8],
    [6.5, 59.6],
    [10.9, 80.2],
    [8.6, 69.7],
    [9.8, 76.5],
    [7.1, 61.8],
    [11.2, 82.8],
    [6.9, 58.1],
    [8.2, 66.2],
    [10.6, 79.6],
    [7.7, 63.4],
    [9.5, 74.9],
    [8.3, 67.8],
    [6.4, 57.5],
    [10.1, 77.1],
    [9.7, 75.8],
    [8.8, 69.2],
    [7.3, 62.7],
    [11.4, 83.4],
    [6.1, 56.3],
    [8.5, 68.6],
    [10.8, 80.9],
    [7.9, 64.1],
    [9.2, 73.5],
    [8.0, 65.4],
    [6.7, 59.8],
    [11.1, 82.2],
    [7.6, 63.9],
    [9.9, 76.1],
    [8.4, 67.3],
    [10.3, 78.7],
    [6.3, 57.7],
    [9.6, 75.2],
    [8.1, 66.8],
    [7.2, 61.5],
    [10.7, 79.4],
    [6.8, 60.1],
    [9.4, 74.6],
    [8.9, 70.8],
    [7.8, 64.5],
    [11.3, 83.0],
    [6.6, 58.9],
    [10.0, 77.6],
    [8.7, 69.4],
    [9.1, 72.1],
    [7.5, 62.3],
    [6.2, 56.6],
    [10.9, 81.1],
    [8.3, 67.9],
    [9.8, 76.3],
    [7.4, 63.7],
    [11.0, 82.5],
    [6.9, 59.2],
    [8.6, 68.8],
    [10.4, 78.4],
    [7.1, 61.0],
    [9.7, 75.7],
    [8.2, 66.5],
    [6.5, 58.2],
    [10.6, 79.9],
    [9.3, 74.2],
    [8.8, 69.9],
    [7.9, 64.6],
    [11.2, 82.7],
    [6.4, 57.4],
    [8.5, 67.6],
    [10.8, 80.8],
    [7.7, 63.1],
    [9.5, 75.0],
    [8.1, 66.1],
    [6.8, 59.5],
    [11.4, 83.6],
    [7.3, 62.8],
    [9.9, 76.9],
    [8.4, 68.2],
    [10.2, 78.0],
    [6.1, 56.0],
    [9.6, 75.5],
    [8.7, 69.1],
    [7.6, 63.3],
    [10.1, 77.2],
    [6.7, 60.7],
    [9.2, 73.8],
    [8.9, 70.5],
    [7.0, 61.4],
    [11.1, 82.1],
    [6.3, 57.1],
    [10.5, 79.3],
    [8.6, 68.4],
    [9.8, 76.6],
    [7.8, 64.9],
    [6.6, 59.0],
    [10.9, 80.6],
    [8.2, 66.7],
    [9.4, 74.4],
    [7.5, 62.5],
    [11.3, 83.2],
    [6.2, 57.8],
    [8.0, 65.2],
    [10.7, 79.7],
    [7.2, 61.7],
    [9.1, 72.4],
    [8.8, 69.8],
    [6.9, 60.4],
    [11.0, 81.9],
    [7.9, 64.0],
    [9.7, 75.6],
    [8.3, 67.0],
    [10.4, 78.5],
    [6.5, 58.6],
    [9.3, 73.1],
    [8.1, 66.3],
    [7.4, 62.2],
    [10.6, 79.0],
    [6.8, 59.7],
    [11.2, 82.3],
    [8.9, 70.2],
    [9.6, 75.3],
    [7.1, 61.1],
    [10.3, 77.8],
    [6.4, 57.6],
    [8.7, 68.7],
    [9.0, 71.5],
    [7.7, 63.6],
    [11.4, 83.5],
    [6.1, 56.4],
    [10.8, 80.4],
    [8.4, 67.4],
    [9.9, 76.0],
    [7.6, 62.9],
    [6.7, 59.4],
    [10.1, 77.5],
    [8.6, 68.9],
    [9.2, 73.2],
    [7.3, 61.9],
    [11.1, 82.6],
    [6.3, 57.3],
    [8.1, 65.8],
    [10.5, 78.8],
    [7.8, 64.2],
    [9.5, 74.7],
    [8.8, 70.0],
    [6.6, 58.8],
    [10.9, 80.1],
    [7.0, 60.6],
    [9.8, 76.4]
])  # (shoe size, weight in kg)

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

y = np.array([162, 168, 164, 174, 178, 163, 170, 161, 176, 182, 158, 169, 165, 180, 160, 173, 171, 162, 181, 159, 168, 175, 164, 183, 163, 167, 177, 164, 175, 170, 161, 180, 162, 173, 168, 182, 163, 167, 178, 165, 175, 160, 166, 179, 164, 174, 167, 183, 161, 176, 162, 170, 179, 163, 174, 169, 165, 181, 159, 177, 167, 172, 165, 161, 180, 170, 176, 162, 182, 161, 166, 179, 165, 175, 168, 160, 177, 176, 170, 164, 183, 159, 169, 180, 165, 173, 166, 161, 182, 165, 176, 168, 178, 160, 175, 167, 162, 179, 163, 174, 171, 165, 183, 161, 177, 170, 172, 164, 159, 181, 168, 176, 165, 182, 162, 169, 178, 162, 176, 167, 161, 179, 174, 170, 165, 182, 160, 168, 180, 164, 175, 167, 162, 183, 164, 176, 168, 178, 159, 175, 170, 165, 177, 163, 173, 171, 162, 181, 160, 179, 169, 176, 165, 161, 180, 167, 174, 164, 183, 160, 166, 179, 162, 172, 170, 163, 181, 165, 176, 167, 178, 161, 173, 167, 164, 179, 163, 182, 170, 175, 162, 178, 160, 169, 171, 165, 183, 159, 180, 168, 176, 164, 162, 177, 169, 173, 163, 182, 160, 166, 179, 165, 175, 170, 161, 180, 162, 176])  # Heights (cm)

m, n = X.shape  # m = number training set, n = number of features


# Initialize weights and bias
W = np.array([1.0, 1.0])# Random values for w1, w2
b = 0.0

# Hyperparameters
learning_rate = 0.00001  # Learning rate (step size)
number_of_iterations = 1000000  # Number of iterations

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





















