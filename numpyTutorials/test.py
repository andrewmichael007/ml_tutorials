#first implementing the cost function
import numpy as np
# m is the number of training examples
# y_hat is the predicted value
# y is the actual value

def cost_function( y_hat, y, m):
    squared_error = (y_hat - y) ** 2
    cost = (1 / (2 * m )) * np.sum(squared_error)
    return cost

y_hat = np.array([3, 4, 2])
y = np.array([2, 5, 1])
m = len(y)

cost = cost_function(m, y_hat, y)
print(cost)
