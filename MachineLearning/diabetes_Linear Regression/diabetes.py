# diabetes.csv

# MACHINE LEARNING

# LINEAR REGRESSION

# IMPORT LIBRARIES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)


# LOAD DATASET

df = pd.read_csv(

    "Data_Sets/Machine Learning_Linear Regression/diabetes.csv"

)


# DATASET INFORMATION

print(df.head())

print(df.info())

print(df.describe())


# REMOVE MISSING VALUES

df = df.dropna()


# FEATURES

X = df.drop(

    columns="Outcome"

)


# TARGET

y = df["Outcome"]


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split( X,y,

    test_size=0.20,

    random_state=42

)


# BUILD MODEL

model = LinearRegression()


# TRAIN MODEL

model.fit( X_train, y_train)


# PREDICTION

y_pred = model.predict( X_test)


# MODEL COEFFICIENTS

print("\nMODEL COEFFICIENTS")

for feature, coefficient in zip(

    X.columns,

    model.coef_

):

    print(f"{feature:30} {coefficient:.6f}")


# INTERCEPT

print("\nINTERCEPT")

print(model.intercept_)


# EVALUATION

mae = mean_absolute_error(

    y_test,

    y_pred

)

mse = mean_squared_error(

    y_test,

    y_pred

)

rmse = np.sqrt(

    mse

)

r2 = r2_score(

    y_test,

    y_pred

)


# RESULTS

print("\nLINEAR REGRESSION RESULTS")

print("MAE :", mae)

print("MSE :", mse)

print("RMSE :", rmse)

print("R² Score :", r2)


# ACTUAL VS PREDICTED

comparison = pd.DataFrame(

    {

        "Actual": y_test.values,

        "Predicted": y_pred

    }

)

print(comparison.head(10))


# ACTUAL VS PREDICTED LINE PLOT

comparison = comparison.reset_index(drop=True)

plt.figure(figsize=(12,6))

plt.plot(

    comparison["Actual"],

    label="Actual",

    linewidth=3

)

plt.plot( comparison["Predicted"], label="Predicted" )

plt.title("Actual vs Predicted Outcome")

plt.xlabel("Observations")

plt.ylabel("Outcome")

plt.legend()

plt.grid(True)

plt.show()


# ACTUAL VS PREDICTED SCATTER PLOT

plt.figure(figsize=(8,6))

plt.scatter(y_test,y_pred)

plt.plot(

    [0,1],

    [0,1],

    "r--"

)

plt.title("Actual vs Predicted Outcome")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.grid(True)

plt.show()


# FEATURE IMPORTANCE

importance = pd.DataFrame(

 {

        "Feature": X.columns,

        "Coefficient": model.coef_

    }

)

importance = importance.sort_values( by="Coefficient",ascending=False)

print("\nFEATURE IMPORTANCE")

print(importance)

