# Advertising.csv

# LINEAR REGRESSION

# IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)

import matplotlib.pyplot as plt


# LOAD DATASET

df = pd.read_csv( "Data_Sets/Machine Learning_Linear Regression/advertising.csv" )


# DATASET INFORMATION

print(df.head())

print(df.info())

print(df.describe())


# FEATURES AND TARGET

X = df[

    [

        "TV",

        "Radio",

        "Newspaper"

    ]

]

y = df["Sales"]


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

print(model.coef_)

print("\nIntercept")

print(model.intercept_)
# EVALUATION

mae = mean_absolute_error( y_test, y_pred )

mse = mean_squared_error( y_test, y_pred)

rmse = np.sqrt( mse )

r2 = r2_score( y_test,  y_pred )


# RESULTS

print("\nLINEAR REGRESSION RESULTS")

print("MAE :", mae)

print("MSE :", mse)

print("RMSE :", rmse)

print("R² Score :", r2)


# ACTUAL VS PREDICTED

comparison = pd.DataFrame(

    {

        "Actual Sales": y_test.values,

        "Predicted Sales": y_pred

    }

)

print(comparison.head(10))


# ACTUAL VS PREDICTED PLOT

plt.figure(figsize=(10,6))

plt.plot(

    y_test.values,

    label="Actual",

    linewidth=3

)

plt.plot(

    y_pred,

    label="Predicted"

)

plt.title("Actual vs Predicted Sales")

plt.xlabel("Observations")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()


# SCATTER PLOT

plt.figure(figsize=(8,6))

plt.scatter( y_test, y_pred )

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    "r--"

)

plt.title("Actual vs Predicted")

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.grid(True)

plt.show()


# FEATURE IMPORTANCE

importance = pd.DataFrame(

    {

        "Feature": X.columns,

        "Coefficient": model.coef_

    }

)

print("\nFeature Importance")

print(importance.sort_values(

    by="Coefficient",

    ascending=False

))


