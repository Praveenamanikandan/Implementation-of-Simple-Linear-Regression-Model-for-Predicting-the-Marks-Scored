# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the necessary libraries for data handling, visualization, and model building.
2.Load the dataset and inspect the first and last few records to understand the data structure.
3.Prepare the data by separating the independent variable (hours studied) and the dependent variable (marks scored).
4.Split the dataset into training and testing sets to evaluate the model's performance.
5.Initialize and train a linear regression model using the training data.
6.Predict the marks for the test set using the trained model.
7.Evaluate the model by comparing the predicted marks with the actual marks from the test set.
8.Visualize the results for both the training and test sets by plotting the actual data points and the regression line
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAVEENA M
RegisterNumber:  212223040153
*/


import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```



## Output:

![Screenshot 2024-09-02 220748](https://github.com/user-attachments/assets/5aef223f-53e9-4a2f-a54f-85b45535a68a)

```
dataset.info()
```

## Output:
![Screenshot 2024-09-02 220902](https://github.com/user-attachments/assets/084b92fb-741e-4fe4-ad90-b39d99969fed)

```

X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```

## Output:
![Screenshot 2024-09-02 221005](https://github.com/user-attachments/assets/dcc27a78-c07a-4470-bb21-a9fc94528529)

```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)


```

## Output:
![Screenshot 2024-09-02 221055](https://github.com/user-attachments/assets/fd15cb82-d477-4773-9df0-e38ee7264042)

```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
## Output:
![Screenshot 2024-09-02 221139](https://github.com/user-attachments/assets/ff21a936-1262-4c44-ad44-88e4e709de78)

```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)

```
## Output:

![Screenshot 2024-09-02 221306](https://github.com/user-attachments/assets/e3ac1e7f-2010-44b3-94b2-610b21218c29)

```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="green")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:

![Screenshot 2024-09-02 221459](https://github.com/user-attachments/assets/4955942c-5de4-4b88-bfab-c1bb1ddf24a7)

```
plt.scatter(X_test, Y_test,color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:

![image](https://github.com/user-attachments/assets/700e0a3a-ddf1-494a-903a-351afd21cce5)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
