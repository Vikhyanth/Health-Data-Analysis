import pandas as pd

# Load dataset
df = pd.read_csv('diabetes.csv')

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure for saving plots
plt.figure(figsize=(12, 6))

# Histogram for Age
plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Number of Patients')

# Histogram for BMI
plt.subplot(1, 2, 2)
sns.histplot(df['BMI'], bins=30, kde=True)
plt.title('BMI Distribution of Patients')
plt.xlabel('BMI')
plt.ylabel('Count of Individuals')

# Save the figure as a PNG file
plt.tight_layout()
plt.savefig('age_bmi_distribution.png')  # Save the figure
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Features and target variable
X = df[['Age', 'BMI']]
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))
import numpy as np

# Create a grid to plot decision boundary
x_min, x_max = X['Age'].min() - 1, X['Age'].max() + 1
y_min, y_max = X['BMI'].min() - 1, X['BMI'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision boundary and training points
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x='Age', y='BMI', hue='Outcome', data=df,
                palette={0: 'blue', 1: 'orange'}, edgecolor='k')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.legend(title='Diabetes Outcome', loc='upper right', labels=['No Diabetes', 'Diabetes'])

# Save this figure as a PNG file
plt.savefig('decision_boundary.png')  # Save the figure
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('diabetes.csv')

# Features and target variable
X = df.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')
y = df['Outcome']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy percentage
print(f'Accuracy of the prediction model: {accuracy * 100:.2f}%')
