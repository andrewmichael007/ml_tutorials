import numpy as np

first_array = np.array([[1,2] , [3,4]])
second_array = np.array([[5,6] , [7,8]])

print(first_array + second_array)

#multiplication
third_array = np.array([[1,2] , [3,4]])
fourth_array = np.array([[5,6] , [7,8]])

print("***********************************")
print(third_array * fourth_array)
print("***********************************")


#finding the transpose
fifth_array = np.array([[1,2] , [3,4]])
print(fifth_array.T)

#working out for the determinant
sixth_array = np.array([[1,2] , [3,4]])
print(np.linalg.det(sixth_array))

