# Starlink & SpaceX Data Dataset_spacex_launches.csv

"""
OBJECTIVE:

The objective of this project is to analyze historical stock market data
using Python, Data Analysis, Visualization, and Deep Learning techniques.

The project begins with NumPy for numerical computations, Pandas for
data preprocessing and feature engineering, and Seaborn for exploratory
data analysis (EDA) to understand price trends, distributions, and
relationships among stock market variables.

After preprocessing the dataset, Time Series forecasting models are
developed using TensorFlow-Keras, including:

• Recurrent Neural Network (RNN)
• Long Short-Term Memory (LSTM)
• Gated Recurrent Unit (GRU)

Each model is trained to predict future stock closing prices based on
historical market data.

Finally, the models are evaluated and compared using standard regression
metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE),
Root Mean Squared Error (RMSE), and R² Score to determine the most
accurate forecasting model.

The overall goal is to identify temporal patterns in stock prices and
build an effective Deep Learning model for stock price prediction.
"""

# IMPORT LIBRARIES

# NumPy is used for numerical computing, mathematical calculations, statistical analysis, and array operations.
import numpy as np

# Pandas is used to load, clean, transform, anipulate, and analyze structured datasets.
import pandas as pd

# Matplotlib is the base plotting library used to create charts, graphs, and figures.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Seaborn is built on top of Matplotlib and is used for creating attractive statistical visualizations.
import seaborn as sns

# MinMaxScaler scales numerical features into the range [0,1], which is recommended for RNN, LSTM, and GRU models.
from sklearn.preprocessing import MinMaxScaler

# Mean Absolute Error (MAE)
# Measures the average absolute prediction error.
from sklearn.metrics import mean_absolute_error

# Mean Squared Error (MSE)
# Measures the average squared prediction error.
from sklearn.metrics import mean_squared_error

# R² Score
# Measures how well the model explains the variance in the target variable.
from sklearn.metrics import r2_score

# TensorFlow is Google's Deep Learning framework.
import tensorflow as tf

# Sequential is used to build neural networks layer-by-layer.
from tensorflow.keras.models import Sequential

# Import Deep Learning layers.
from tensorflow.keras.layers import (SimpleRNN,LSTM,GRU,Dense,Dropout)

# Adam optimizer is widely used for training deep learning models.
from tensorflow.keras.optimizers import Adam

# EarlyStopping stops training automatically when validation performance stops improving.
from tensorflow.keras.callbacks import EarlyStopping

# Display all DataFrame columns.
pd.set_option("display.max_columns", None)

# Display DataFrame width completely.
pd.set_option("display.width", None)

# Display floating-point values with 2 decimal places.
pd.options.display.float_format = "{:.2f}".format

# Set a professional plotting style.
sns.set_style("whitegrid")

# Set default figure size for all plots.
plt.rcParams ["figure.figsize"] = (10, 6)

# Fix NumPy random seed so results remain reproducible.
np.random.seed(42)

# Fix TensorFlow random seed so model training is reproducible.
tf.random.set_seed(42)


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load the CSV dataset using NumPy.

success,failures, crew_count, payloads_count,cores_reused, landing_success, = np.genfromtxt("Data_Sets/Starlink & SpaceX Data Dataset/spacex_launches.csv", delimiter=",", usecols=(7,8,10,11,12,13), unpack=True, dtype=float,skip_header=1 , filling_values=np.nan)


print(success)

print(failures)

print(crew_count)

print(payloads_count)

print(cores_reused)

print(landing_success)

print("Total Launches :", len(success))

print("Shape :", success.shape)

print("Dimensions :", success.ndim)

print("Data Type :", success.dtype)

# Missing Values

print("Missing Success :", np.isnan(success).sum())

print("Missing Failures :", np.isnan(failures).sum())

print("Missing Crew :", np.isnan(crew_count).sum())

print("Missing Payload :", np.isnan(payloads_count).sum())

print("Missing Reused Cores :", np.isnan(cores_reused).sum())

print("Missing Landing :", np.isnan(landing_success).sum())

# Statistical Analysis of Success

print(np.min(success))

print(np.max(success))

print(np.mean(success))

print(np.median(success))

print(np.std(success))

print(np.var(success))

# Failures

print(np.min(failures))

print(np.max(failures))

print(np.mean(failures))

print(np.median(failures))

print(np.std(failures))

# Crew

print(np.min(crew_count))

print(np.max(crew_count))

print(np.mean(crew_count))

print(np.median(crew_count))

# Payload Count

print(np.min(payloads_count))

print(np.max(payloads_count))

print(np.mean(payloads_count))

print(np.median(payloads_count))

# Reused Cores

print(np.min(cores_reused))

print(np.max(cores_reused))

print(np.mean(cores_reused))

# Total Values

print("Total Success =", np.nansum(success))

print("Total Failures =", np.nansum(failures))

print("Total Crew =", np.nansum(crew_count))

print("Total Payloads =", np.nansum(payloads_count))

print("Total Reused Cores =", np.nansum(cores_reused))

print("Successful Landings =", np.nansum(landing_success))

# Count Values

print("Successful Launches =", np.count_nonzero(success==1))

print("Failed Launches =", np.count_nonzero(success==0))

print("Landing Success =", np.count_nonzero(landing_success==1))

print("Landing Failure =", np.count_nonzero(landing_success==0))

# Successful Launches

successful = success[success==1]

print(successful)

# Failed Launches

failed = failures[failures>0]

print(failed)

# Missions Carrying Crew

crew = crew_count[crew_count>0]

print(crew)

# Missions Carrying Multiple Payloads

multiple_payload = payloads_count[payloads_count>1]

print(multiple_payload)

# Reused Rockets

reused = cores_reused[cores_reused>0]

print(reused)

# Logical Relations
# Launch Success vs Landing Success

both = np.sum((success==1) & (landing_success==1))

print("Successful Launch + Successful Landing =", both)

# Crew Missions that were Successful

crew_success = np.sum((crew_count>0) & (success==1))

print(crew_success)

# Payload Missions that Failed

payload_failure = np.sum((payloads_count>0) & (success==0))

print(payload_failure)

# Reused Rockets that Successfully Landed

reuse_land = np.sum((cores_reused>0) & (landing_success==1))

print(reuse_land)

# Reused Rockets with Successful Launch

reuse_success = np.sum((cores_reused>0) & (success==1))

print(reuse_success)

# Average Payload in Successful Missions

avg_payload = np.mean(payloads_count[success==1])

print(avg_payload)

# Average Crew in Successful Missions

avg_crew = np.mean(crew_count[success==1])

print(avg_crew)

# Average Reused Core in Successful Missions

avg_reuse = np.mean(cores_reused[success==1])

print(avg_reuse)

# Missions Having Crew and Multiple Payloads

crew_payload = np.sum((crew_count>0) & (payloads_count>1))

print(crew_payload)

# Correlation Analysis (NumPy)

data = np.array([
    success,
    failures,
    crew_count,
    payloads_count,
    cores_reused,
    landing_success
])

correlation = np.corrcoef(data)

print(correlation)


# Pandas
# Load the SpaceX Dataset

df = pd.read_csv("Data_Sets/Starlink & SpaceX Data Dataset/spacex_launches.csv")

print("Dataset Loaded Successfully")

# Display first 5 rows
print("\nFirst 5 Rows")

print(df.head())

# Display last 5 rows
print("\nLast 5 Rows")

print(df.tail())

# Random 5 rows
print("\nRandom Sample")

print(df.sample(5, random_state=42))

# DATASET INFORMATION

print("\nShape of Dataset")

print(df.shape)

print("\nNumber of Rows :", df.shape[0])

print("Number of Columns :", df.shape[1])

print("\nColumn Names")

print(df.columns.tolist())

print("\nData Types")

print(df.dtypes)

print("\nDataset Information")

print(df.info())

# STATISTICS

print("\nNumerical Statistics")

print(df.describe())

print("\nCategorical Statistics")

print(df.describe(include='object'))

# MISSING VALUES ANALYSIS

print("\nMissing Values")

print(df.isnull().sum())

print("\nTotal Missing Values :", df.isnull().sum().sum())

print("\nMissing Value Percentage")

missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage.sort_values(ascending=False))

# DUPLICATE RECORD ANALYSIS

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

# Remove duplicate rows if present

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates Removed Successfully")
else:
    print("No Duplicate Records Found")


# UNIQUE VALUES

print("\nUnique Values in Each Column")
print(df.nunique())


# MEMORY USAGE

print("\nMemory Usage")

print(df.memory_usage(deep=True))

print("\nTotal Memory Used (MB):")

print(df.memory_usage(deep=True).sum()/1024**2)


# COLUMN-WISE ANALYSIS

for column in df.columns:
    print("="*60)
    print("Column :", column)
    print("Data Type :", df[column].dtype)
    print("Unique Values :", df[column].nunique())

    # Show top values
    print(df[column].value_counts().head())

    print()

numerical_columns = df.select_dtypes(include=['int64','float64']).columns

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"{column} : {len(outliers)} Outliers")

# Convert launch date into datetime format

df["date_utc"] = pd.to_datetime(df["date_utc"])

print(df["date_utc"].head())

df["Year"] = df["date_utc"].dt.year
df["Month"] = df["date_utc"].dt.month
df["Day"] = df["date_utc"].dt.day
df["Weekday"] = df["date_utc"].dt.day_name()

print(df[["date_utc","Year","Month","Day","Weekday"]].head())

# DATA CLEANING
# Fill numerical missing values with median

numerical_columns = df.select_dtypes(include=['number']).columns

for col in numerical_columns: df[col].fillna(df[col].median(), inplace=True)


# Fill categorical missing values with mode

categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns: df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing Values After Cleaning")

print(df.isnull().sum())


#SAVE CLEAN DATASET

df.to_csv("Cleaned_SpaceX_Dataset.csv", index=False)

print("Clean Dataset Saved Successfully")


# Monthly Launches

monthly_launches = df.groupby("Month").size().reset_index(name="Launch Count")

print(monthly_launches)

# Launches by Weekday

weekday_launches = df["Weekday"].value_counts()

print(weekday_launches)

# Success Rate by Year

success_year = ( df.groupby("Year")["success"].mean().reset_index(name="Average Success Rate") )

print(success_year)


# Average Payload Per Year

payload_year = ( df.groupby("Year")["payloads_count"].mean().reset_index(name="Average Payload") )

print(payload_year)

# Crew Missions Per Year

crew_year = ( df.groupby("Year")["crew_count"].sum().reset_index(name="Total Crew") )

print(crew_year)

# Rocket-wise Analysis
# Rocket Performance


rocket_analysis = ( df.groupby("rocket_name") .agg(
          Total_Launches=("rocket_name", "count"),
          Success_Rate=("success", "mean"),
          Average_Payload=("payloads_count", "mean")
      )
      .sort_values(by="Success_Rate", ascending=False)
)

print(rocket_analysis)


# Launch Site Performance

site_analysis = ( df.groupby("launchpad_name").agg( Launches=("launchpad_name", "count"), Success=("success", "mean") )
      .sort_values(by="Launches", ascending=False)
)

print(site_analysis)


# Landing Success

landing = df["landing_success"].value_counts()
print(landing)

# Core Reuse Analysis

reuse = ( df.groupby("cores_reused").agg( Missions=("cores_reused", "count"), Success=("success", "mean") ) )

print(reuse)

# Top 10 Missions with Highest Payload
# Highest Payload Missions

top_payload = ( df.sort_values("payloads_count", ascending=False).head(10) )

print(top_payload)


# Successful Missions

successful = df[df["success"] == 1]

print("Successful Missions :", len(successful))


# Failed Missions

failed = df[df["success"] == 0]

print("Failed Missions :", len(failed))


# Human Missions

human = df[df["crew_count"] > 0]

print(human.head())

print("Total Human Missions :", len(human))


# Cargo Missions

cargo = df[df["crew_count"] == 0]

print("Cargo Missions :", len(cargo))

# Pivot Table (Year vs Success)
# Pivot Table

pivot = pd.pivot_table( df, values="success", index="Year", aggfunc="mean" )

print(pivot)


# Correlation Matrix

correlation = df.select_dtypes(include="number").corr()

print(correlation)

# Feature Engineering
# Human Mission

df["Human_Mission"] = (df["crew_count"] > 0).astype(int)

# Reused Booster

df["Reused_Core"] = (df["cores_reused"] > 0).astype(int)

# Heavy Payload Mission

df["Heavy_Payload"] = ( df["payloads_count"] > df["payloads_count"].median() ).astype(int)

print(df[["Human_Mission", "Reused_Core", "Heavy_Payload"]].head())


# SEABORN DATA VISUALIZATION

plt.figure(figsize=(10,6))

sns.countplot(data=df, x="success")

plt.title("Mission Success Distribution")

plt.xlabel("Mission Success")

plt.ylabel("Number of Missions")

plt.show()

# PAYLOAD COUNT DISTRIBUTION

plt.figure(figsize=(10,6))

sns.histplot(df["payloads_count"], kde=True)

plt.title("Payload Count Distribution")

plt.xlabel("Payload Count")

plt.ylabel("Frequency")

plt.show()

# CREW COUNT DISTRIBUTION

plt.figure(figsize=(10,6))

sns.histplot(df["crew_count"], kde=True)

plt.title("Crew Count Distribution")

plt.xlabel("Crew Count")

plt.ylabel("Frequency")

plt.show()

# CORE REUSE DISTRIBUTION

plt.figure(figsize=(10,6))

sns.histplot(df["cores_reused"], kde=True)

plt.title("Core Reuse Distribution")

plt.xlabel("Reused Cores")

plt.ylabel("Frequency")

plt.show()

# LANDING SUCCESS DISTRIBUTION

plt.figure(figsize=(10,6))

sns.countplot(data=df, x="landing_success")

plt.title("Landing Success Distribution")

plt.xlabel("Landing Success")

plt.ylabel("Number of Missions")

plt.show()

# ROCKET-WISE MISSIONS

plt.figure(figsize=(12,6))

sns.countplot(data=df, x="rocket_name")

plt.title("Rocket-wise Missions")

plt.xlabel("Rocket")

plt.ylabel("Launch Count")

plt.xticks(rotation=45)

plt.show()

# LAUNCHPAD-WISE MISSIONS

plt.figure(figsize=(12,6))

sns.countplot(data=df, x="launchpad_name")

plt.title("Launchpad-wise Missions")

plt.xlabel("Launchpad")

plt.ylabel("Launch Count")

plt.xticks(rotation=45)

plt.show()

# LANDING TYPE DISTRIBUTION

plt.figure(figsize=(12,6))

sns.countplot(data=df, x="landing_type")

plt.title("Landing Type Distribution")

plt.xlabel("Landing Type")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.show()

# PAYLOAD VS SUCCESS

plt.figure(figsize=(10,6))

sns.boxplot(data=df, x="success", y="payloads_count")

plt.title("Payload Count by Mission Success")

plt.xlabel("Mission Success")

plt.ylabel("Payload Count")

plt.show()

# CREW VS SUCCESS

plt.figure(figsize=(10,6))

sns.boxplot(data=df, x="success", y="crew_count")

plt.title("Crew Count by Mission Success")

plt.xlabel("Mission Success")

plt.ylabel("Crew Count")

plt.show()

# CORE REUSE VS LANDING SUCCESS

plt.figure(figsize=(10,6))

sns.boxplot(data=df, x="landing_success", y="cores_reused")

plt.title("Core Reuse by Landing Success")

plt.xlabel("Landing Success")

plt.ylabel("Reused Cores")

plt.show()

# PAYLOAD VS CREW

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="payloads_count",
    y="crew_count",
    hue="success"
)

plt.title("Payload Count vs Crew Count")

plt.xlabel("Payload Count")

plt.ylabel("Crew Count")

plt.show()

# CORE REUSE VS PAYLOAD

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="cores_reused",
    y="payloads_count",
    hue="landing_success"
)

plt.title("Core Reuse vs Payload Count")

plt.xlabel("Reused Cores")

plt.ylabel("Payload Count")

plt.show()

# CREW DISTRIBUTION BY SUCCESS

plt.figure(figsize=(10,6))

sns.violinplot(data=df, x="success", y="crew_count")

plt.title("Crew Distribution by Mission Success")

plt.xlabel("Mission Success")

plt.ylabel("Crew Count")

plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(10,8))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# PAIR PLOT

sns.pairplot(
    df[
        [
            "success",
            "failures",
            "crew_count",
            "payloads_count",
            "cores_reused",
            "landing_success"
        ]
    ]
)

plt.show()

# YEARLY LAUNCH TREND

launches = df.groupby("Year").size().reset_index(name="Launches")

plt.figure(figsize=(10,6))

sns.lineplot(
    data=launches,
    x="Year",
    y="Launches",
    marker="o"
)

plt.title("Launch Trend by Year")

plt.xlabel("Year")

plt.ylabel("Number of Launches")

plt.show()

# YEARLY SUCCESS RATE

success_rate = df.groupby("Year")["success"].mean().reset_index()

plt.figure(figsize=(10,6))

sns.lineplot(
    data=success_rate,
    x="Year",
    y="success",
    marker="o"
)

plt.title("Mission Success Rate by Year")

plt.xlabel("Year")

plt.ylabel("Average Success Rate")

plt.show()

# ROCKET SUCCESS RATE

rocket_success = df.groupby("rocket_name")["success"].mean().reset_index()

plt.figure(figsize=(12,6))

sns.barplot(
    data=rocket_success,
    x="rocket_name",
    y="success"
)

plt.title("Average Mission Success by Rocket")

plt.xlabel("Rocket")

plt.ylabel("Average Success Rate")

plt.xticks(rotation=45)

plt.show()


# LAUNCHPAD SUCCESS RATE

launchpad_success = df.groupby("launchpad_name")["success"].mean().reset_index()

plt.figure(figsize=(12,6))

sns.barplot(
    data=launchpad_success,
    x="launchpad_name",
    y="success"
)

plt.title("Average Mission Success by Launchpad")

plt.xlabel("Launchpad")

plt.ylabel("Average Success Rate")

plt.xticks(rotation=45)

plt.show()


# FEATURE Engineering 
#  HUMAN MISSION

# 1 = Human Mission
# 0 = Cargo Mission

df["Human_Mission"] = (df["crew_count"] > 0).astype(int)

print(df[["crew_count", "Human_Mission"]].head())

# REUSED BOOSTER

# 1 = Reused Booster
# 0 = New Booster

df["Reused_Booster"] = (df["cores_reused"] > 0).astype(int)

print(df[["cores_reused", "Reused_Booster"]].head())

# HEAVY PAYLOAD

median_payload = df["payloads_count"].median()

df["Heavy_Payload"] = ( df["payloads_count"] >= median_payload ).astype(int)

print(df[["payloads_count", "Heavy_Payload"]].head())

# SUCCESSFUL LANDING

df["Successful_Landing"] = ( df["landing_success"] == 1 ).astype(int)

print(df[["landing_success", "Successful_Landing"]].head())

# FAILURE CATEGORY

df["Failure_Category"] = df["failures"].apply( lambda x: "Failure" if x > 0 else "No Failure" )

print(df[["failures", "Failure_Category"]].head())


# LAUNCH QUARTER

df["Quarter"] = df["date_utc"].dt.quarter

print(df[["Month", "Quarter"]].head())

# MISSION COMPLEXITY

df["Mission_Complexity"] = (
    df["crew_count"] +
    df["payloads_count"] +
    df["cores_reused"]
)

print(df[
    [
        "crew_count",
        "payloads_count",
        "cores_reused",
        "Mission_Complexity"
    ]
].head())

# CORRELATION ANALYSIS

# CORRELATION MATRIX

correlation_matrix = df.select_dtypes(include="number").corr()

print(correlation_matrix)

# CORRELATION HEATMAP

plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Matrix")

plt.show()

# CORRELATION WITH SUCCESS

success_correlation = ( correlation_matrix["success"].sort_values(ascending=False) )

print(success_correlation)

# TOP POSITIVE CORRELATIONS

print(success_correlation.head(10))

# TOP NEGATIVE CORRELATIONS

print(success_correlation.tail(10))


# SPECIFIC FEATURE CORRELATIONS

print( "Payload vs Success :", df["payloads_count"].corr(df["success"]) )

print( "Crew vs Success :", df["crew_count"].corr(df["success"]) )

print( "Core Reuse vs Landing Success :", df["cores_reused"].corr(df["landing_success"]) )

print( "Failures vs Success :", df["failures"].corr(df["success"]) )


# SAVE UPDATED DATASE

df.to_csv( "SpaceX_Feature_Engineered.csv", index=False )

print("Feature Engineered Dataset Saved Successfully.")



# Classification

# Create a copy of dataset

classification_df = df.copy()

# Convert categorical columns into numerical values

encoder = LabelEncoder()

categorical_columns = classification_df.select_dtypes(include="object").columns

for column in categorical_columns:

    classification_df[column] = encoder.fit_transform(
        classification_df[column].astype(str)
    )

# DEFINE FEATURES AND TARGET
# Independent Variables

X = classification_df.drop(
    columns=[
        "success",             # Target
        "date_utc",            # Timestamp
        "landing_success",     # Potential leakage
        "Successful_Landing"   # Derived from landing_success
    ],
    errors="ignore"
)

y = classification_df["success"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split( X, y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# CREATE MODELS

models = {

    "Logistic Regression": LogisticRegression(),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(random_state=42),

    "KNN": KNeighborsClassifier(),

    "SVM": SVC()

}

# TRAIN & EVALUATE MODELS

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append([name, accuracy])

    print("=" * 60)

    print(name)

    print("=" * 60)

    print("Accuracy :", round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")

    print(classification_report(y_test, predictions)
    
)

    
# MODEL COMPARISON

results_df = pd.DataFrame( results, columns=[ "Model",  "Accuracy" ] )

results_df = results_df.sort_values( by="Accuracy", ascending=False )

print(results_df)

# CLASSIFICATION VISUALIZATION

plt.figure(figsize=(10,6))

sns.barplot( data=results_df, x="Model", y="Accuracy" )

plt.title("Classification Model Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.show()

# Best Model

best_model = results_df.iloc[0]

print("="*60)

print("Best Classification Model")

print(best_model)

print("="*60)


# REGRESSION

# DATA PREPARATION

regression_df = df.copy()

# Encode categorical columns
encoder = LabelEncoder()

categorical_columns = regression_df.select_dtypes(include="object").columns

for column in categorical_columns:

    regression_df[column] = encoder.fit_transform( regression_df[column].astype(str) )

# Remove columns that should not be used
regression_df.drop(
    columns=["date_utc"],
    errors="ignore",
    inplace=True
)

# Features
X = regression_df.drop(columns=["payloads_count"])

# Target
y = regression_df["payloads_count"]


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split( X, y,

    test_size=0.20,

    random_state=42
)


# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# REGRESSION MODELS

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(random_state=42),

    "Random Forest": RandomForestRegressor(random_state=42)

}

# MODEL TRAINING

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)

    mse = mean_squared_error(y_test, prediction)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, prediction)

    results.append([name, mae, mse, rmse, r2])

    print("="*60)

    print(name)

    print("="*60)

    print("MAE  :", mae)

    print("MSE  :", mse)

    print("RMSE :", rmse)

    print("R² Score :", r2)

  
# MODEL COMPARISON

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "MAE",

        "MSE",

        "RMSE",

        "R2 Score"

    ]

)

print(results_df)


# REGRESSION MODEL COMPARISON

plt.figure(figsize=(10,6))

sns.barplot(

    data=results_df,

    x="Model",

    y="R2 Score"

)

plt.title("Regression Model Comparison")

plt.xlabel("Model")

plt.ylabel("R² Score")

plt.show()


# CLUSTERING

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

# SELECT NUMERICAL FEATURES

cluster_df = df.copy()

cluster_features = cluster_df[

    [

        "crew_count",

        "payloads_count",

        "cores_reused",

        "failures"

    ]

]

# SCALE FEATURES

scaler = StandardScaler()

cluster_scaled = scaler.fit_transform(cluster_features)

# ELBOW METHOD


wcss = []

for i in range(1,11):

    kmeans = KMeans(

        n_clusters=i,

        random_state=42,

        n_init=10

    )

    kmeans.fit(cluster_scaled)

    wcss.append(kmeans.inertia_)

# Plot Elbow Method 

plt.figure(figsize=(10,6))

plt.plot(

    range(1,11),

    wcss,

    marker="o"

)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.show()


# KMEANS

kmeans = KMeans(

    n_clusters=3,

    random_state=42,

    n_init=10

)

clusters = kmeans.fit_predict(cluster_scaled)

cluster_df["Cluster"] = clusters


# SILHOUETTE SCORE

score = silhouette_score(

    cluster_scaled,

    clusters

)

print("Silhouette Score :", score)


# CLUSTER COUNT

print(cluster_df["Cluster"].value_counts())

# CLUSTER VISUALIZATION

plt.figure(figsize=(10,6))

sns.scatterplot(

    data=cluster_df,

    x="payloads_count",

    y="crew_count",

    hue="Cluster",

    palette="Set2",

    s=80

)

plt.title("Mission Clusters")

plt.xlabel("Payload Count")

plt.ylabel("Crew Count")

plt.show()


# CLUSTER SUMMARY

summary = cluster_df.groupby("Cluster")[

    [

        "crew_count",

        "payloads_count",

        "cores_reused",

        "failures"

    ]

].mean()

print(summary)

# TIME SERIES FORECASTING
# PREPARE TIME SERIES

time_df = df.copy()

time_df["date_utc"] = pd.to_datetime(time_df["date_utc"])

monthly_launches = (

    time_df

    .set_index("date_utc")

    .resample("MS")

    .size()

    .to_frame(name="Launch_Count")

)

print(monthly_launches.head())


# VISUALIZE TIME SERIES

plt.figure(figsize=(12,6))

plt.plot(

    monthly_launches.index,

    monthly_launches["Launch_Count"],

    marker="o"

)

plt.title("Monthly SpaceX Launch Count")

plt.xlabel("Date")

plt.ylabel("Launch Count")

plt.grid(True)

plt.show()


# NORMALIZATION

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(monthly_launches)


# CREATE SEQUENCES

sequence_length = 6

X = []

y = []

for i in range(sequence_length, len(scaled_data)):

    X.append(scaled_data[i-sequence_length:i])

    y.append(scaled_data[i])

X = np.array(X)

y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

print("X Shape :", X.shape)

print("y Shape :", y.shape)


# TRAIN TEST SPLIT

split = int(len(X) * 0.80)

X_train = X[:split]

X_test = X[split:]

y_train = y[:split]

y_test = y[split:]

print("Training Samples :", len(X_train))

print("Testing Samples :", len(X_test))


# TRAIN AND EVALUATE MODEL

def train_deep_learning_model(layer, model_name):

    model = Sequential()

    model.add(

        layer(

            units=64,

            activation="tanh",

            input_shape=(X_train.shape[1], X_train.shape[2])

        )

    )

    model.add(Dense(1))

    model.compile(

        optimizer="adam",

        loss="mse"

    )

    history = model.fit(

        X_train,

        y_train,

        epochs=50,

        batch_size=8,

        validation_split=0.20,

        verbose=1

    )

    prediction = model.predict(

        X_test,

        verbose=0

    )

    prediction = scaler.inverse_transform(prediction)

    actual = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(actual, prediction)

    mse = mean_squared_error(actual, prediction)

    rmse = np.sqrt(mse)

    print("\n" + "="*60)

    print(model_name)

    print("="*60)

    print("MAE :", round(mae,4))

    print("MSE :", round(mse,4))

    print("RMSE :", round(rmse,4))

    return model, history, prediction, mae, mse, rmse


# TRAIN SIMPLE RNN

rnn_model, rnn_history, rnn_prediction, rnn_mae, rnn_mse, rnn_rmse = train_deep_learning_model(

    SimpleRNN,

    "Simple RNN"

)


# TRAIN LSTM

lstm_model, lstm_history, lstm_prediction, lstm_mae, lstm_mse, lstm_rmse = train_deep_learning_model(

    LSTM,

    "LSTM"

)


# TRAIN GRU

gru_model, gru_history, gru_prediction, gru_mae, gru_mse, gru_rmse = train_deep_learning_model(

    GRU,

    "GRU"

)


# MODEL COMPARISON

comparison = pd.DataFrame({

    "Model":[

        "Simple RNN",

        "LSTM",

        "GRU"

    ],

    "MAE":[

        rnn_mae,

        lstm_mae,

        gru_mae

    ],

    "MSE":[

        rnn_mse,

        lstm_mse,

        gru_mse

    ],

    "RMSE":[

        rnn_rmse,

        lstm_rmse,

        gru_rmse

    ]

})

print(comparison)


# BEST MODEL

best_model = comparison.loc[

    comparison["RMSE"].idxmin()

]

print("\nBest Deep Learning Model")

print(best_model)


# ACTUAL VS PREDICTIONS

actual = scaler.inverse_transform(y_test)

plt.figure(figsize=(12,6))

plt.plot(

    actual,

    label="Actual",

    linewidth=3

)

plt.plot(

    rnn_prediction,

    label="Simple RNN"

)

plt.plot(

    lstm_prediction,

    label="LSTM"

)

plt.plot(

    gru_prediction,

    label="GRU"

)

plt.title("Actual vs Predicted Monthly Launch Count")

plt.xlabel("Time")

plt.ylabel("Launch Count")

plt.legend()

plt.grid(True)

plt.show()


# TRAINING LOSS

plt.figure(figsize=(12,6))

plt.plot(

    rnn_history.history["loss"],

    label="Simple RNN"

)

plt.plot(

    lstm_history.history["loss"],

    label="LSTM"

)

plt.plot(

    gru_history.history["loss"],

    label="GRU"

)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# VALIDATION LOSS

plt.figure(figsize=(12,6))

plt.plot(

    rnn_history.history["val_loss"],

    label="Simple RNN"

)

plt.plot(

    lstm_history.history["val_loss"],

    label="LSTM"

)

plt.plot(

    gru_history.history["val_loss"],

    label="GRU"

)

plt.title("Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Validation Loss")

plt.legend()

plt.grid(True)

plt.show()


print("\n" + "="*70)
print("FINAL PROJECT CONCLUSION")
print("="*70)

print("""
This project successfully implemented an end-to-end Data Science and Deep
Learning workflow using the SpaceX Launch Dataset.

The project included:

✓ NumPy for numerical analysis.
✓ Pandas for data preprocessing and feature engineering.
✓ Seaborn & Matplotlib for data visualization.
✓ Correlation analysis to identify feature relationships.
✓ Machine Learning models for Classification, Regression, and Clustering.
✓ Deep Learning models (Simple RNN, LSTM, and GRU) for time series forecasting.

Model performance was evaluated using Accuracy, R² Score, MAE, MSE, RMSE,
and Silhouette Score. The best-performing models were selected based on
their evaluation metrics.

Overall, this project demonstrates practical skills in data preprocessing,
visualization, machine learning, clustering, and deep learning, providing
valuable insights into SpaceX launch operations and future launch trends.
""")