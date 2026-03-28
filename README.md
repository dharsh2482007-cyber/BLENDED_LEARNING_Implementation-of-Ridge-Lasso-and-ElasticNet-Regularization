# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset, then preprocess it by converting categorical variables into numerical form.
 

2.Split the dataset into training and testing sets and standardize the feature values.


3.Create pipelines with Polynomial Features and apply Ridge, Lasso, and ElasticNet regression models.


4.Train the models, predict car prices, and evaluate their performance using MSE and R² score.

  
 

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: P.PRIYADHARSHINI
RegisterNumber: 212225220076
*/


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("CarPrice_Assignment.csv")
data.head()
data=pd.get_dummies(data,drop_first=True)
# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']
# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))
# Splitting the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, rando
# Define the models and pipelines 
models = {"Ridge": Ridge(alpha=1.0),
"Lasso": Lasso(alpha=1.0),
"ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
         }
# Dictionary to store results 
results = {}

# Train and evaluate each model 
for name, model in models.items():
# Create a pipeline with polynomial features and the model 
pipeline = Pipeline([('poly', PolynomialFeatures(degree=2)), ('regressor',
# Fit the model 
pipeline.fit (X_train, y_train)
# Make predictions
predictions = pipeline.predict(X_test)
# Calculate performance metrics
mse = mean_squared_error(y_test, predictions) 
r2 = r2_score(y_test, predictions)
#Store results
results [name] = {'MSE': mse, 'R² Score': r2}
# Print results
print('Name:PRIYADHARSHINI')
print('Reg. No: 212225220076')
for model_name, metrics in results.items():
print(f"{model_name} - Mean Squared Error:{metrics ['MSE']:.2f}, R² Score:
#Visualization of the results
#Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
#Set the figure size
plt.figure(figsize=(12, 5))
# Bar plot for MSE
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
# Bar plot for R² Score
plt.subplot(1, 2, 2) 
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis') 
plt.title('R² Score')
plt.ylabel('R² Score')
plt.xticks (rotation=45)
# Show the plots 
plt.tight_layout() 
plt.show()
Name:PRIYADHARSHINI
Reg. No: 212225220076
Ridge - Mean Squared Error:1.47, R² Score: -0.18
Lasso - Mean Squared Error:1.29, R² Score: -0.04
ElasticNet - Mean Squared Error:1.33, R² Score: -0.07

```

## Output:
<img width="834" height="253" alt="Screenshot 2026-03-28 212150" src="https://github.com/user-attachments/assets/205acb8a-773a-462d-b3bf-325f54ee090e" />
<img width="802" height="148" alt="Screenshot 2026-03-28 212204" src="https://github.com/user-attachments/assets/d9281e25-92f0-4c22-8f02-f9d63f715102" />
<img width="395" height="127" alt="Screenshot 2026-03-28 212224" src="https://github.com/user-attachments/assets/2a7fcf73-8311-42a1-9a6f-d92a6393a9c8" />
<img width="503" height="658" alt="Screenshot 2026-03-28 212239" src="https://github.com/user-attachments/assets/4ff5df8a-721c-4edc-9a8b-ca4fe720aef9" />
<img width="436" height="673" alt="Screenshot 2026-03-28 212249" src="https://github.com/user-attachments/assets/813f7a0b-e4ee-4adb-a9f2-b1168e7097db" />



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
