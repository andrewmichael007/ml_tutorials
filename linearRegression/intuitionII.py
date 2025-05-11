# from intuition.py, we manually updated the w and b component.
# In this program we try to update w and b automatically using an optimization algorithm gradient descent


# GRADIENT DESCENT INTUITION
import numpy as np
import matplotlib.pyplot as plt

# Data: shoe sizes and heights
shoe_sizes = np.array([7, 8, 9, 10, 11])  # feature (x)
heights = np.array([160, 165, 170, 175, 180])  # target (y))
m = len(heights)  # Number of  training set

# setting w and b to initial values or initialize w (slope) and b (intercept) randomly
w = 4.51 # np.random.rand()
b = 128  # np.random.rand()

# Learning rate (step size)
alpha = 0.000009
number_of_iterations = 1000  # Number of iterations

# store cost history for plotting
cost_history = []

# gradient descent algorithm
for each in range(number_of_iterations):
    # compute prediction for each value of w, b for y_hat
    y_hat = w * shoe_sizes + b
    
    # compute error
    errors = np.sum(y_hat - heights)

    # this block of code computes the mean squareed error

    squared_error  = (errors ** 2)# for squarred error
    cost_function =  (1 / (2 * m)) * (squared_error) # final cost function calculation
    cost_history.append(cost_function)

    # the next block of code updates the w and b parameters automatically
    # the algorithm;
    # for w; 
    # 1. take the initial value of  w
    # 2. compute their derivative, how ? just multipy  1/m by the sum of the errors
    # 3. now multiply the derivative by alpha
    # 4. subtract the total from the w
    # 5. total is the new value  of w

    derivative_of_w = (1 / m) * np.sum(errors * shoe_sizes)  
    derivative_of_b = (1 / m) * np.sum(errors) 

    # updating the  parameters of w and b
    w -= alpha * derivative_of_w
    b -= alpha * derivative_of_b

    # print cost every 100 iterations
    if each % 100 == 0:
        print( f"Iteration {each}: Cost = {cost_function}, w = {w:.4f}, b = {b:.4f}" )


# Final values of w and b after training
print("\nFinal w:", w)
print("Final b:", b)

# Plot cost function convergence
plt.plot(range(number_of_iterations), cost_history, '-b', linewidth=2)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.show()

# Final predictions
y_hat = w * shoe_sizes + b

# Plot predictions vs actual values
plt.scatter(shoe_sizes, heights, color="blue", label="real heights")
plt.plot(shoe_sizes, y_hat, color="red", label="predicted heights")
plt.xlabel("Shoe Size")
plt.ylabel("Height (cm)")
plt.legend()
plt.title("Final Model Fit")
plt.show()


