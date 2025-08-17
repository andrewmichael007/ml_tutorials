# Simple linear model to predict heights based on the shoe size

# importing various libraries.
import numpy as np
import matplotlib.pyplot as plt

# mocked  data for now
#input feature x
shoe_sizes = np.array([
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16 , 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52
]) 

# input feature y
heights = np.array([
    150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 
    205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 
    280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 
    355, 360, 365, 370, 375
]) 


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

m = len(heights) # number of training set
cost_function =  (1 / (2 * m )) * (errors ** 2)
print("COST FUNCTION: " , cost_function)

#plot
plt.scatter(shoe_sizes, heights, color = "blue", label = "actual height ") #this is the actual height
plt.plot(shoe_sizes , y_hat , color = "red" , label = "prediction height ") #this is the predicted height
plt.xlabel("Shoe Size")
plt.ylabel("Height (cm)")
plt.legend()
plt.show()



def predict_height(shoe_size, w, b):
    return w * shoe_size + b

# Example usage
shoe_size = 53
predicted_height = predict_height(shoe_size, w, b)
print(f"The predicted height for a shoe size of {shoe_size} is {predicted_height:.2f} cm")
