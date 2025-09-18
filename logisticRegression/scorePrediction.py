# importing various models
import numpy as np
import matplotlib.pyplot as plt

# dummy dataset: exam1 score, exam2 score -> pass/fail
# input
X = np.array([
    [45, 52],   # fail
    [38, 41],   # fail
    [55, 47],   # fail
    [42, 39],   # fail
    [51, 58],   # fail
    [48, 45],   # fail
    [44, 50],   # fail
    [39, 43],   # fail
    [53, 49],   # fail
    [46, 44],   # fail
    [41, 38],   # fail
    [47, 51],   # fail
    [50, 46],   # fail
    [43, 42],   # fail
    [49, 53],   # fail
    [52, 48],   # fail
    [40, 45],   # fail
    [54, 50],   # fail
    [45, 47],   # fail
    [48, 44],   # fail
    [37, 40],   # fail
    [56, 52],   # fail
    [44, 46],   # fail
    [49, 48],   # fail
    [42, 43],   # fail
    [51, 49],   # fail
    [47, 45],   # fail
    [53, 51],   # fail
    [46, 42],   # fail
    [50, 47],   # fail
    [41, 44],   # fail
    [48, 50],   # fail
    [45, 43],   # fail
    [52, 46],   # fail
    [49, 45],   # fail
    [43, 48],   # fail
    [54, 49],   # fail
    [47, 44],   # fail
    [51, 47],   # fail
    [46, 50],   # fail
    [40, 42],   # fail
    [55, 48],   # fail
    [48, 46],   # fail
    [44, 49],   # fail
    [50, 43],   # fail
    [42, 45],   # fail
    [53, 47],   # fail
    [49, 50],   # fail
    [45, 44],   # fail
    [47, 46],   # fail
    [65, 72],   # pass
    [68, 75],   # pass
    [72, 78],   # pass
    [75, 81],   # pass
    [69, 73],   # pass
    [71, 76],   # pass
    [66, 70],   # pass
    [73, 79],   # pass
    [67, 74],   # pass
    [70, 77],   # pass
    [74, 80],   # pass
    [68, 72],   # pass
    [72, 75],   # pass
    [69, 76],   # pass
    [71, 78],   # pass
    [66, 73],   # pass
    [75, 82],   # pass
    [67, 71],   # pass
    [73, 77],   # pass
    [70, 74],   # pass
    [68, 79],   # pass
    [72, 73],   # pass
    [69, 75],   # pass
    [74, 78],   # pass
    [67, 72],   # pass
    [71, 76],   # pass
    [66, 74],   # pass
    [73, 80],   # pass
    [70, 75],   # pass
    [68, 77],   # pass
    [72, 79],   # pass
    [69, 74],   # pass
    [75, 76],   # pass
    [67, 73],   # pass
    [71, 78],   # pass
    [74, 81],   # pass
    [66, 72],   # pass
    [73, 75],   # pass
    [70, 77],   # pass
    [68, 74],   # pass
    [72, 80],   # pass
    [69, 76],   # pass
    [75, 78],   # pass
    [67, 75],   # pass
    [71, 73],   # pass
    [74, 79],   # pass
    [66, 77],   # pass
    [73, 74],   # pass
    [70, 78],   # pass
    [68, 76],   # pass
])

# output
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # labels (0 = fail, 1 = pass)

# Visualize - we check whether  through the ouput labels, if it's 0 then label is failed else if 1, pass
for each in range(len(y)):
    if y[each] == 0:
        plt.scatter(X[each,0], X[each,1], color="red", marker="x", label="Fail" if each == 0 else "")
    else:
        plt.scatter(X[each,0], X[each,1], color="blue", marker="o", label="Pass" if each == 3 else "")

#sigmoid function to calculate the probabilities
def sigmoid (z):
    return 1 / (1 + np.exp(-z))

#hypothesis function
def predict_prob(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

# cost function (log loss)
def compute_cost(X, y, w, b):
    m = len(y)
    h = predict_prob(X, w, b)
    cost_function = - (1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return cost_function

# gredient descent
def gradient_descent(X, y, w, b, alpha, iterations):
    m = len(y)
    cost_history = []

    for each in range(iterations):
        h = predict_prob(X, w, b)
        dw = (1/m) * np.dot(X.T, (h - y))
        db = (1/m) * np.sum(h - y)

        w -= alpha * dw
        b -= alpha * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if each % 100 == 0:
            print(f"Iteration {each}, Cost: {cost:.4f}")

    return w, b, cost_history


# Initialize parameters
w = np.zeros(X.shape[1])
b = 0
alpha = 0.01
iterations = 1000

w, b, cost_history = gradient_descent(X, y, w, b, alpha, iterations)

print("Final weights:", w)
print("Final bias:", b)

#visualization of the graph
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.title("Student Exam Scores - Pass/Fail")
plt.show()
