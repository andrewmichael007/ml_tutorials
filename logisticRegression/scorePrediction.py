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
for i in range(len(y)):
    if y[i] == 0:
        plt.scatter(X[i,0], X[i,1], color="red", marker="x", label="Fail" if i==0 else "")
    else:
        plt.scatter(X[i,0], X[i,1], color="blue", marker="o", label="Pass" if i==3 else "")




#sigmoid function to calculate the probabilities
def sigmoid (z):
    return 1 / (1 + np.exp(-z))

#hypothesis function
def predict_probability(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

def compute_cost(X, y, w, b):
    m = len(y)
    h = predict_proba(X, w, b)
    cost = - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    return cost



plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.title("Student Exam Scores - Pass/Fail")
plt.show()
