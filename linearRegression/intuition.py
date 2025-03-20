# Simple linear model to predict shoe sizes based on the height

# importing various libraries.
import numpy as np
import matplotlib.pyplot as plt

# Fake data for now
shoe_sizes = np.array([7, 8, 9, 10, 11]) #input 
heights = np.array([160, 165, 170, 175, 180])

# y_hat is the predictions

# w and b are parameters

# x is an input feature, in this case it's the shoe sizes

# w and b are the  slope and intercept respectively
# w, b = 5 , 130    w and b for intuition1.png
# w, b = 4 , 128  w and for intuition2.png
w, b = 4.51 , 128

# formula for the function (predictions)
y_hat = w * shoe_sizes + b

# the function is making predictions
print("PREDICTIONS: ", y_hat , "\n")

# Error
errors = float(np.sum(y_hat - heights))
print("ERROR: ", errors, "\n")

# Squared Error
squared_error =  (errors ** 2)
print("SQUARED ERROR: ",  squared_error, "\n")

# MEAN SQUARED ERROR FUNCTION / COST FUNCTION
# calculating the difference between the actual and predicted values to know how wrong we are.

m = len(heights)
cost_function =  (1 / (2 * m )) * (errors ** 2)
print("COST FUNCTION: " , cost_function)

#PLOT
plt.scatter(shoe_sizes, heights, color = "blue", label = "actual height ")
plt.plot(shoe_sizes , y_hat , color = "red" , label = "prediction height ")
plt.xlabel("Shoe Size")
plt.ylabel("Height (cm)")
plt.legend()
plt.show()

