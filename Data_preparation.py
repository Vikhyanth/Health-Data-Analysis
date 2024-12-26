import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Ensure this line is included to avoid NameError
from scipy import stats
# Load the dataset
df = pd.read_csv('diabetes.csv')

# Display the first few rows of the dataset
print(df.head())

missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Example: Filling missing values with the mean for numerical columns
df.fillna(df.mean(), inplace=True)

# Alternatively, drop rows with missing values if necessary
# df.dropna(inplace=True)
# Descriptive statistics
print(df.describe())









