# importing various models
import numpy as np
import matplotlib.pyplot as plt

# dummy dataset: exam1 score, exam2 score -> pass/fail
# input
X = np.array([
    [35, 67],   # fail
    [45, 78],   # fail
    [50, 43],   # fail
    [65, 80],   # pass
    [72, 60],   # pass
    [88, 95],   # pass
])

# output
y = np.array([0, 0, 0, 1, 1, 1])  # labels (0 = fail, 1 = pass)

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



plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.title("Student Exam Scores - Pass/Fail")
plt.show()
