import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('diabetes.csv')

# Data exploration (check for missing values, basic statistics)
print(df.head())
print(df.isnull().sum())
print(df.describe())

# Hypothesis Testing Example: T-test for BMI between diabetic and non-diabetic groups
group1 = df[df['Outcome'] == 1]['BMI']
group0 = df[df['Outcome'] == 0]['BMI']
t_stat, p_value = stats.ttest_ind(group1, group0)
print(f'T-statistic: {t_stat}, P-value: {p_value}')

# Logistic Regression Analysis
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']                 # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
