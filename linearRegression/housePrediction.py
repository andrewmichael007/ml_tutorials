# this model predicts housing prices using the  california housing dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# import seaborn as sns

# feature scaling (meanin normalization + standardization)
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Loading the California housing dataset
california_housing_data = fetch_california_housing()

# Creating DataFrame for features and Series for target variable
# features are like the inputs
features = pd.DataFrame(california_housing_data.data, columns=california_housing_data.feature_names)

# target are the outputs
target = pd.Series(california_housing_data.target, name="HouseValue" )

# Exploring the data
#data preview
print("features(x):\n", features.head())

# rows, cols
print("\nShape of features: ", features.shape)

# rows
print("Shape of target", target.shape)

scaler = StandardScaler()

# Scale the features (turning big numbers into manageable ones)
features_scaled = scaler.fit_transform(features)

# Add a column of ones to act as x₀
# Number of examples
m = features_scaled.shape[0]

# Final input matrix
X = np.column_stack((np.ones(m), features_scaled))

# Convert dataframe to NumPy array
y = target.values


# Cost function
def compute_cost(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)  # Predictions
    errors = predictions -y
    squarred_error = errors ** 2
    cost_function = (1 / (2 * m)) * np.sum(squarred_error)
    return cost_function


# Gradient Descent Algorithm
def gradient_descent(X, y, learning_rate, number_of_iterations):
    m = len(y)
    cost_history = []

    num_features = X.shape[1]
    weights = np.zeros(num_features)  # Initialize weights

    for each in range(number_of_iterations):
        y_hat = X.dot(weights)  # Predictions

        # Calculate errors
        errors = y_hat - y # Difference between predictions and actual values

        # Compute gradients
        gradients = (1 / m) * X.T.dot(errors)

        # Update weights
        weights -= learning_rate * gradients

        # Compute cost and store it
        cost = compute_cost(X, y, weights) #calling the compute_cost function
        cost_history.append(cost)

        # Print cost every 100 iterations
        if each % 100 == 0:
            print(f"Iteration {each}: Cost = {cost:.4f}, Weights = {weights}")

    return weights, cost_history

# Initial weights (0s)
num_features = X.shape[1]
initial_weights = np.zeros(num_features)

# Set learning rate and number of steps
learning_rate = 0.005
number_of_iterations = 1000

# Train the model
final_weights, cost_list = gradient_descent(X, y, learning_rate, number_of_iterations)

print("\nTraining done.")
print("\nFinal weights (parameters): \n", final_weights)
print("\nFinal cost:", cost_list[-1])

#Plotting cost history
plt.plot(range(number_of_iterations), cost_list, color='purple')
plt.title("Cost Function Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.grid(True)
plt.show()

