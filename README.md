# Diabetes Analysis Project

## Overview
This project aims to analyze the diabetes dataset to understand the relationship between various health metrics (such as Age and BMI) and the likelihood of diabetes. The analysis includes data exploration, visualization, and the implementation of a machine learning model to predict diabetes outcomes based on these metrics.

## Dataset
The dataset used for this analysis is `diabetes.csv`, which contains the following columns:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1) indicating whether diabetes is present

## Steps Taken

### 1. Environment Setup
- Created a GitHub repository and opened it in GitHub Codespaces.
- Installed necessary Python libraries including `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.

### 2. Data Loading and Exploration
- Loaded the dataset using Pandas and performed initial exploration to understand its structure.
- Checked for missing values and basic statistics.

### 3. Data Preparation
- Selected relevant features for analysis, focusing on `Age`, `BMI`, and `Outcome`.
- Split the dataset into training and testing sets for model evaluation.

### 4. Exploratory Data Analysis (EDA)
- Created histograms to visualize the distribution of Age and BMI.
- Generated scatter plots to explore relationships between Age, BMI, and diabetes outcomes.

### 5. Machine Learning Implementation
- Implemented a Logistic Regression model to predict diabetes outcomes based on Age and BMI.
- Evaluated the model's performance using accuracy metrics.

### 6. Visualization of Results
- Plotted decision boundaries to visualize how well the model predicts diabetes based on Age and BMI.
- Saved visualizations as PNG files for documentation.

## Model Accuracy
The accuracy of the prediction model was calculated using the test dataset. The final accuracy percentage indicates how well the model can predict diabetes outcomes based on input features.

## Conclusion
This project demonstrates the ability to analyze health data, perform exploratory data analysis, implement machine learning models, and visualize results effectively. The insights gained can help in understanding risk factors associated with diabetes.

## Future Work
Future enhancements could include:
- Exploring additional features from the dataset.
- Implementing more complex machine learning models for improved predictions.
- Conducting a deeper analysis of other health metrics that may influence diabetes risk.

## Files Included
- `diabetes.csv`: The dataset used for analysis.
- `diabetes_analysis.py`: The main script containing all code for data processing, analysis, and visualization.
