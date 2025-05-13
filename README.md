# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SRIRANJANI.M
RegisterNumber:  212224040327
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("Salary.csv")
print(data.head()) 
print(data.info())          
print(data.isnull().sum())
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())
x = data[["Position", "Level"]]  # Features
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)
print("Predicted Salary for [5,6]:", dt.predict([[5, 6]]))
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

<img width="495" alt="01" src="https://github.com/user-attachments/assets/9fd13ae2-a47a-47a3-b74c-aa311c2960e3" />

<img width="554" alt="02" src="https://github.com/user-attachments/assets/c694e117-9f4f-4e19-8cc4-b1bbf1b810e2" />

<img width="615" alt="03" src="https://github.com/user-attachments/assets/f1e63663-e69a-4eec-88d0-76a62e69afce" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
