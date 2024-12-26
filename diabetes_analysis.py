import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('diabetes.csv')

# Display first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Get basic statistics of the dataset
print(df.describe())

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Number of Patients', fontsize=14)  # Updated label
plt.grid(axis='y')
plt.savefig('age_distribution.png')  # Save the plot as an image file
plt.show()

# BMI Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['BMI'], bins=30, kde=True)
plt.title('BMI Distribution', fontsize=16)
plt.xlabel('BMI', fontsize=14)
plt.ylabel('Count of Individuals', fontsize=14)  # Updated label
plt.grid(axis='y')
plt.savefig('bmi_distribution.png')  # Save the plot as an image file
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.ylabel('Features', fontsize=14)  # Added label for clarity
plt.savefig('correlation_heatmap.png')  # Save the plot as an image file
plt.show()

# Outcome Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Outcome', data=df)
plt.title('Diabetes Distribution', fontsize=16)
plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)', fontsize=14)
plt.ylabel('Count of Cases', fontsize=14)  # Updated label
plt.grid(axis='y')
plt.savefig('outcome_distribution.png')  # Save the plot as an image file
plt.show()
