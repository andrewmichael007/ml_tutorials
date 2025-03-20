import numpy as np
import matplotlib.pyplot as plt

# loading the  data set
data = np.loadtxt("../datasets/house_prices_dataset.csv", delimiter=",", skiprows=1)


X = data[:, 0] #feature
y = data[:, 0] #target
m = len(y) # number of training examples

print(X[:5] , y[:5]) #checking the first five rows to verify
# X = np.column_stack((np.ones(m), X))

# visualizing the dataset
plt.scatter(data[:, 0], data[:, 1], color = "red", marker = "x")
plt.xlabel("Size of House (sq ft)")
plt.ylabel("Price (USD)")
plt.title("Housing Prices")
plt.show()


