# Boston.csv

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

df = pd.read_csv( "Data_Sets/Machine Learning_Linear Regression/Boston.csv" )


# DATASET INFORMATION

print(df.head())

print(df.info())

print(df.describe())


# REMOVE MISSING VALUES

df = df.dropna()


# FEATURES

X = df.drop( columns="medv" )


# TARGET

y = df["medv"]


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split( X, y,

    test_size=0.20,

    random_state=42

)


# BUILD MODEL

model = LinearRegression()


# TRAIN MODEL

model.fit( X_train, y_train )


# PREDICTIONS

y_pred = model.predict( X_test )


# MODEL COEFFICIENTS

print("\nModel Coefficients")

for feature, coefficient in zip( X.columns, model.coef_): print(feature, ":", coefficient)


# INTERCEPT

print("\nIntercept")

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


# ACTUAL VS PREDICTED TABLE

comparison = pd.DataFrame(

    {

        "Actual Price": y_test.values,

        "Predicted Price": y_pred

    }

)

print(comparison.head(10))


# ACTUAL VS PREDICTED LINE PLOT

comparison = comparison.reset_index( drop=True )

plt.figure(figsize=(12,6))

plt.plot(

    comparison["Actual Price"],

    label="Actual",

    linewidth=3

)

plt.plot(

    comparison["Predicted Price"],

    label="Predicted"

)

plt.title("Actual vs Predicted House Prices")

plt.xlabel("Observations")

plt.ylabel("Median House Value")

plt.legend()

plt.grid(True)

plt.show()


# ACTUAL VS PREDICTED SCATTER PLOT

plt.figure(figsize=(8,6))

plt.scatter(

    y_test,

    y_pred

)

plt.plot(

    [ y_test.min(), y_test.max() ],

    [ y_test.min(), y_test.max() ],

    "r--"

)

plt.title("Actual vs Predicted House Prices")

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

importance = importance.sort_values(

    by="Coefficient",

    ascending=False

)

print("\nFeature Importance")

print(importance)