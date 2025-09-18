# loading and exploring dataset

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset (UCI Energy Efficiency)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"

# reading excel file directly into pandas
data_frame = pd.read_excel(url)

# inspect dataset
print("Dataset shape:", data_frame.shape)
print(data_frame.head())

# renaming columns for readability
data_frame.columns = [
    'Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
    'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution',
    'Heating Load', 'Cooling Load'
]

# identifying features (X) and target (y)
X = data_frame.iloc[:, :-2]   # all columns except last two
y = data_frame['Heating Load']   # our target variable

# visualization
sns.pairplot(data_frame[['Relative Compactness', 'Surface Area', 'Wall Area', 'Heating Load']])
plt.show()
