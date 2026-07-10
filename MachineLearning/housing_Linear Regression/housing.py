# housing.csv

# MACHINE LEARNING

# LINEAR REGRESSION

# IMPORT LIBRARIES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)


# LOAD DATASET

df = pd.read_csv( "Data_Sets/Machine Learning_Linear Regression/housing.csv" )


# DATASET INFORMATION

print(df.head())

print(df.info())

print(df.describe())


# MISSING VALUES

print(df.isnull().sum())


# DUPLICATES

print("Duplicate Rows :", df.duplicated().sum())


# REMOVE DUPLICATES

df = df.drop_duplicates()


# FEATURES

X = df.drop( columns="median_house_value" )


# TARGET

y = df["median_house_value"]


# NUMERICAL FEATURES

numeric_features = [

    "longitude",

    "latitude",

    "housing_median_age",

    "total_rooms",

    "total_bedrooms",

    "population",

    "households",

    "median_income"

]


# CATEGORICAL FEATURES

categorical_features = ["ocean_proximity" ]


# PREPROCESSOR

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            SimpleImputer(strategy="median"),

            numeric_features

        ),

        (

            "cat",

            OneHotEncoder(handle_unknown="ignore"),

            categorical_features

        )

    ]

)


# BUILD PIPELINE

model = Pipeline(

    steps=[

        (

            "preprocessor",

            preprocessor

        ),

        (

            "regressor",

            LinearRegression()

        )

    ]

)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split( X, y,

    test_size=0.20,

    random_state=42

)


# TRAIN MODEL

model.fit(   X_train,   y_train )


# PREDICTIONS

y_pred = model.predict( X_test )


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

print("MAE  :", mae)

print("MSE  :", mse)

print("RMSE :", rmse)

print("R² Score :", r2)


# ACTUAL VS PREDICTED TABLE

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

plt.plot(

    comparison["Predicted"],

    label="Predicted"

)

plt.title("Actual vs Predicted House Value")

plt.xlabel("Observations")

plt.ylabel("Median House Value")

plt.legend()

plt.grid(True)

plt.show()


# ACTUAL VS PREDICTED SCATTER PLOT

plt.figure(figsize=(8,6))

plt.scatter( y_test, y_pred, alpha=0.6 )

plt.plot(

    [

        y_test.min(),

        y_test.max()

    ],

    [

        y_test.min(),

        y_test.max()

    ],

    "r--"

)

plt.title("Actual vs Predicted")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.grid(True)

plt.show()


# RESIDUALS

residuals = y_test - y_pred


# RESIDUAL PLOT

plt.figure(figsize=(8,6))

plt.scatter(

    y_pred,

    residuals,

    alpha=0.6

)

plt.axhline(

    y=0,

    color="red",

    linestyle="--"

)

plt.title("Residual Plot")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.grid(True)

plt.show()


# MODEL COEFFICIENTS

regressor = model.named_steps["regressor"]

encoder = model.named_steps["preprocessor"]

feature_names = encoder.get_feature_names_out()

importance = pd.DataFrame(

    {

        "Feature": feature_names,

        "Coefficient": regressor.coef_

    }

)

importance = importance.sort_values( by="Coefficient", ascending=False )

print("\nFEATURE IMPORTANCE")

print(importance)


# INTERCEPT

print("\nINTERCEPT")

print(regressor.intercept_)