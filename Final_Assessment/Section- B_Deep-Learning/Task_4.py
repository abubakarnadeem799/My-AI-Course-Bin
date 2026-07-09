# SpaceX Missions, 2006-Present/database.csv

# IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    SimpleRNN,
    LSTM,
    GRU
)

# NUMPY ANALYSIS

# Load Required Columns using NumPy

# Load Vehicle Type

vehicle_type = np.genfromtxt( "Data_Sets/SpaceX Missions, 2006-Present/database.csv", delimiter=",", usecols=4, skip_header=1, dtype=str )

# Load Payload Mass

payload_mass = np.genfromtxt( "Data_Sets/SpaceX Missions, 2006-Present/database.csv", delimiter=",", usecols=7, skip_header=1, dtype=float, filling_values=np.nan )


print("Shape :", payload_mass.shape)

print("Size :", payload_mass.size)

print("Dimensions :", payload_mass.ndim)

print("Mean :", np.nanmean(payload_mass))

print("Maximum :", np.nanmax(payload_mass))

print("Minimum :", np.nanmin(payload_mass))

print("Vehicle Types")

print(np.unique(vehicle_type))


# PANDAS ANALYSIS
# Load Dataset

df = pd.read_csv("Data_Sets/SpaceX Missions, 2006-Present/database.csv")      

# Convert Payload Mass to Numeric

df["Payload Mass (kg)"] = pd.to_numeric(

    df["Payload Mass (kg)"],

    errors="coerce"

)

# Display First 5 Records

print(df.head())


# Dataset Information

print(df.info())


# Check Missing Values

print("\nMissing Values")

print(df.isnull().sum())


# Check Duplicate Records

print("\nDuplicate Records :", df.duplicated().sum())


# Convert Launch Date to DateTime

df["Launch Date"] = pd.to_datetime(df["Launch Date"])


# Descriptive Statistics

print("\nDataset Statistics")

print(df.describe())


# Group By Vehicle Type

vehicle_analysis = df.groupby("Vehicle Type").agg({ "Payload Mass (kg)" : "mean" })

print("\nAverage Payload Mass by Vehicle Type")

print(vehicle_analysis)


# Group By Mission Outcome

mission_analysis = df.groupby("Mission Outcome").agg({ "Payload Mass (kg)" : ["count","mean"] })

print("\nMission Outcome Analysis")

print(mission_analysis)

# SEABORN DATA VISUALIZATION

# MISSION OUTCOME COUNT PLOT
'''
plt.figure(figsize=(8,5))

sns.countplot(data=df, x="Mission Outcome")

plt.title("Mission Outcome Distribution")

plt.xticks(rotation=30)

plt.show()

# VEHICLE TYPE COUNT PLOT

plt.figure(figsize=(10,5))

sns.countplot(data=df, x="Vehicle Type")

plt.title("Vehicle Type Distribution")

plt.xticks(rotation=45)

plt.show()

# PAYLOAD MASS HISTOGRAM

plt.figure(figsize=(8,5))

sns.histplot(df["Payload Mass (kg)"], kde=True)

plt.title("Payload Mass Distribution")

plt.xlabel("Payload Mass (kg)")

plt.show()

# PAYLOAD MASS BOX PLOT

plt.figure(figsize=(8,5))

sns.boxplot(y=df["Payload Mass (kg)"])

plt.title("Payload Mass Box Plot")

plt.show()

# PAYLOAD MASS VIOLIN PLOT

plt.figure(figsize=(8,5))

sns.violinplot(y=df["Payload Mass (kg)"])

plt.title("Payload Mass Violin Plot")

plt.show()

# LAUNCHES BY YEAR

df["Launch Year"] = df["Launch Date"].dt.year

yearly_launches = df.groupby("Launch Year").size().reset_index(name="Launch Count")

plt.figure(figsize=(10,5))

sns.barplot(data=yearly_launches, x="Launch Year", y="Launch Count")

plt.title("Launches by Year")

plt.xticks(rotation=45)

plt.show()

# PAYLOAD MASS VS VEHICLE TYPE

plt.figure(figsize=(10,5))

sns.scatterplot(
    data=df,
    x="Vehicle Type",
    y="Payload Mass (kg)"
)

plt.title("Payload Mass vs Vehicle Type")

plt.xticks(rotation=45)

plt.show()

# CORRELATION HEATMAP

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

plt.show()

# MONTHLY LAUNCH TREND

df["Launch Month"] = df["Launch Date"].dt.to_period("M").astype(str)

monthly_launches = df.groupby("Launch Month").size().reset_index(name="Launch Count")

plt.figure(figsize=(12,5))

sns.lineplot(
    data=monthly_launches,
    x="Launch Month",
    y="Launch Count",
    marker="o"
)

plt.title("Monthly Launch Trend")

plt.xticks(rotation=90)

plt.show()

# LANDING OUTCOME COUNT PLOT

plt.figure(figsize=(8,5))

sns.countplot(data=df, x="Landing Outcome")

plt.title("Landing Outcome Distribution")

plt.xticks(rotation=30)

plt.show()

# MISSING VALUES HEATMAP

plt.figure(figsize=(10,5))

sns.heatmap(

    df.isnull(),

    cbar=False,

    cmap="viridis"

)

plt.title("Missing Values Heatmap")

plt.show()

# PAYLOAD TYPE COUNT PLOT

plt.figure(figsize=(10,5))

sns.countplot(

    data=df,

    x="Payload Type"

)

plt.title("Payload Type Distribution")

plt.xticks(rotation=45)

plt.show()

# CUSTOMER TYPE COUNT PLOT

plt.figure(figsize=(8,5))

sns.countplot(

    data=df,

    x="Customer Type"

)

plt.title("Customer Type Distribution")

plt.xticks(rotation=30)

plt.show()

# PAYLOAD ORBIT COUNT PLOT

plt.figure(figsize=(10,5))

sns.countplot(

    data=df,

    x="Payload Orbit"

)

plt.title("Payload Orbit Distribution")

plt.xticks(rotation=45)

plt.show()

# PAYLOAD MASS BY MISSION OUTCOME

plt.figure(figsize=(8,5))

sns.boxplot(

    data=df,

    x="Mission Outcome",

    y="Payload Mass (kg)"

)

plt.title("Payload Mass by Mission Outcome")

plt.show()

# PAYLOAD MASS BY VEHICLE TYPE

plt.figure(figsize=(10,5))

sns.violinplot(

    data=df,

    x="Vehicle Type",

    y="Payload Mass (kg)"

)

plt.title("Payload Mass by Vehicle Type")

plt.xticks(rotation=45)

plt.show()

# CUSTOMER COUNTRY DISTRIBUTION

plt.figure(figsize=(12,5))

sns.countplot(

    data=df,

    x="Customer Country",

    order=df["Customer Country"].value_counts().index

)

plt.title("Customer Country Distribution")

plt.xticks(rotation=90)

plt.show()

# PAIR PLOT

sns.pairplot( df.select_dtypes(include="number") )

plt.show() 

'''

# FEATURE ENGINEERING

# Create Launch Year

df["Launch Year"] = df["Launch Date"].dt.year

# Create Launch Month

df["Launch Month"] = df["Launch Date"].dt.month

# Create Mission Success

df["Mission Success"] = df["Mission Outcome"].apply(

    lambda x: 1 if str(x).strip().lower() == "success" else 0

)

# Display Updated Dataset

print(df[["Launch Date",
          "Launch Year",
          "Launch Month",
          "Mission Outcome",
          "Mission Success"]]
          .head()
          
     )

# CORRELATION ANALYSIS

# Select Numerical Columns

correlation_data = df[[
    "Payload Mass (kg)",
    "Launch Year",
    "Mission Success"
]]

# Correlation Matrix

correlation_matrix = correlation_data.corr()

print("Correlation Matrix")

print(correlation_matrix)


# Correlation Heatmap

plt.figure(figsize=(6,4))

sns.heatmap(

    correlation_matrix,

    annot=True,

    cmap="coolwarm",

    linewidths=0.5

)

plt.title("Correlation Heatmap")

plt.show()

# CLASSIFICATION

classification_df = df.copy()

print("Dataset Shape :", classification_df.shape)

# Select Features and Target

X = classification_df[[

    "Vehicle Type",

    "Payload Type",

    "Payload Orbit",

    "Customer Type",

    "Launch Year",

    "Launch Month"

]].copy()

y = classification_df["Mission Outcome"].fillna("Unknown")


# Handle Missing Values

for column in X.columns:

    if pd.api.types.is_numeric_dtype(X[column]):

        X[column] = X[column].fillna(

            X[column].median()

        )

    else:

        X[column] = X[column].fillna(

            "Unknown"

        )

# Encode Categorical Features

encoder = LabelEncoder()

for column in X.select_dtypes(include="object").columns:

    X[column] = encoder.fit_transform(X[column].astype(str))


# Encode Target

y = encoder.fit_transform(y)


# Feature Scaling

scaler = StandardScaler()

X = scaler.fit_transform(X)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.30,

    random_state=42

)

# TRAIN CLASSIFICATION MODEL

def train_classifier(model, model_name):

    model.fit(

        X_train,

        y_train

    )

    prediction = model.predict(X_test)

    accuracy = accuracy_score(

        y_test,

        prediction

    )

    precision = precision_score(

        y_test,

        prediction,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_test,

        prediction,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_test,

        prediction,

        average="weighted",

        zero_division=0

    )

    print("\n" + "="*50)

    print(model_name)

    print("="*50)

    print("Accuracy :", round(accuracy,4))

    print("Precision :", round(precision,4))

    print("Recall :", round(recall,4))

    print("F1 Score :", round(f1,4))

    return accuracy, precision, recall, f1

    # LOGISTIC REGRESSION

log_accuracy, log_precision, log_recall, log_f1 = train_classifier(

    LogisticRegression(

        max_iter=1000,

        random_state=42

    ),

    "Logistic Regression"

)

# RANDOM FOREST

rf_accuracy, rf_precision, rf_recall, rf_f1 = train_classifier(

    RandomForestClassifier(

        n_estimators=100,

        random_state=42

    ),

    "Random Forest"

)

# MODEL COMPARISON

classification_results = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Random Forest"

    ],

    "Accuracy":[

        log_accuracy,

        rf_accuracy

    ],

    "Precision":[

        log_precision,

        rf_precision

    ],

    "Recall":[

        log_recall,

        rf_recall

    ],

    "F1 Score":[

        log_f1,

        rf_f1

    ]

})

print("\nClassification Results")

print(classification_results)


# BEST CLASSIFICATION MODEL

best_model = classification_results.loc[

    classification_results["Accuracy"].idxmax()

]

print("\nBest Classification Model")

print(best_model)

# REGRESSION

# Create Copy

regression_df = df.copy()


# Convert Payload Mass to Numeric

regression_df["Payload Mass (kg)"] = pd.to_numeric(

    regression_df["Payload Mass (kg)"],

    errors="coerce"

)


# Select Features

X = regression_df[[
    "Vehicle Type",
    "Payload Type",
    "Payload Orbit",
    "Customer Type",
    "Launch Year",
    "Launch Month"
]].copy()


# Target

y = regression_df["Payload Mass (kg)"]


# Combine Features and Target

regression_data = pd.concat([X, y], axis=1)


# Remove Missing Values

regression_data = regression_data.dropna().reset_index(drop=True)


# Separate Features and Target

X = regression_data.drop(columns=["Payload Mass (kg)"])

y = regression_data["Payload Mass (kg)"]


# Encode Categorical Features

encoder = LabelEncoder()

for column in X.select_dtypes(include="object").columns:  X[column] = encoder.fit_transform(X[column].astype(str))


# Feature Scaling

scaler = StandardScaler()

X = scaler.fit_transform(X)


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split( X, y,

    test_size=0.30,

    random_state=42

)

# TRAIN REGRESSION MODEL

def train_regressor(model, model_name):

    model.fit(

        X_train,

        y_train

    )

    prediction = model.predict(X_test)

    mae = mean_absolute_error(

        y_test,

        prediction

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            prediction

        )

    )

    r2 = r2_score(

        y_test,

        prediction

    )

    print("\n" + "="*50)

    print(model_name)

    print("="*50)

    print("MAE :", round(mae,2))

    print("RMSE :", round(rmse,2))

    print("R² Score :", round(r2,4))

    return mae, rmse, r2

    # LINEAR REGRESSION

linear_mae, linear_rmse, linear_r2 = train_regressor(

    LinearRegression(),

    "Linear Regression"

)

# RANDOM FOREST REGRESSOR

forest_mae, forest_rmse, forest_r2 = train_regressor(

    RandomForestRegressor(

        n_estimators=100,

        random_state=42

    ),

    "Random Forest Regressor"

)

# MODEL COMPARISON

regression_results = pd.DataFrame({

    "Model":[

        "Linear Regression",

        "Random Forest"

    ],

    "MAE":[

        linear_mae,

        forest_mae

    ],

    "RMSE":[

        linear_rmse,

        forest_rmse

    ],

    "R² Score":[

        linear_r2,

        forest_r2

    ]

})

print("\nRegression Results")

print(regression_results)

# BEST REGRESSION MODEL

best_regression = regression_results.sort_values( by="RMSE" ).iloc[0]

print("\nBest Regression Model")

print(best_regression)

# CLUSTERING

# Create Copy

clustering_df = df.copy()


# Select Features

X = clustering_df[[
    "Vehicle Type",
    "Payload Type",
    "Payload Orbit",
    "Customer Type",
    "Launch Year",
    "Launch Month"
]].copy()


# Handle Missing Values

for column in X.columns:

    X[column] = X[column].fillna("Unknown")


# Encode Categorical Features

encoder = LabelEncoder()

for column in X.columns:

    X[column] = encoder.fit_transform(X[column])


# Feature Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ELBOW METHOD

wcss = []

for k in range(1,11):

    kmeans = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )

    kmeans.fit(X_scaled)

    wcss.append(

        kmeans.inertia_

    )


plt.figure(figsize=(8,5))

plt.plot(

    range(1,11),

    wcss,

    marker="o"

)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.show()

# K-MEANS CLUSTERING

kmeans = KMeans(

    n_clusters=3,

    random_state=42,

    n_init=10

)

clusters = kmeans.fit_predict(

    X_scaled

)

clustering_df["Cluster"] = clusters

# SILHOUETTE SCORE

score = silhouette_score(

    X_scaled,

    clusters

)

print("Silhouette Score :", round(score,4))

# CLUSTER VISUALIZATION

plt.figure(figsize=(8,6))

sns.scatterplot(

    x=X_scaled[:,0],

    y=X_scaled[:,1],

    hue=clusters,

    palette="Set2",

    s=80

)

plt.title("K-Means Cluster Visualization")

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.legend(title="Cluster")

plt.show()



# TIME SERIES

# Create Copy

time_df = df.copy()


# Convert Launch Date

time_df["Launch Date"] = pd.to_datetime(

    time_df["Launch Date"]

)


# Monthly Launch Count

monthly_launches = (

    time_df

    .set_index("Launch Date")

    .resample("MS")

    .size()

    .to_frame(name="Launch Count")

)

print(monthly_launches.head())

# VISUALIZE TIME SERIES

plt.figure(figsize=(12,5))

plt.plot(

    monthly_launches.index,

    monthly_launches["Launch Count"],

    marker="o"

)

plt.title("Monthly Launch Count")

plt.xlabel("Date")

plt.ylabel("Launch Count")

plt.grid(True)

plt.show()

# NORMALIZATION

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(

    monthly_launches

)

# CREATE SEQUENCES

sequence_length = 6

X = []

y = []

for i in range(sequence_length, len(scaled_data)):

    X.append(

        scaled_data[i-sequence_length:i]

    )

    y.append(

        scaled_data[i]

    )

X = np.array(X)

y = np.array(y)

X = X.reshape(

    X.shape[0],

    X.shape[1],

    1

)

print(X.shape)

print(y.shape)

# TRAIN TEST SPLIT

split = int(

    len(X) * 0.80

)

X_train = X[:split]

X_test = X[split:]

y_train = y[:split]

y_test = y[split:]

# TRAIN DEEP LEARNING MODEL

def train_model(layer, model_name):

    model = Sequential()

    model.add(

        layer(

            32,

            input_shape=(

                X_train.shape[1],

                X_train.shape[2]

            )

        )

    )

    model.add(

        Dense(1)

    )

    model.compile(

        optimizer="adam",

        loss="mse"

    )

    history = model.fit(

        X_train,

        y_train,

        epochs=30,

        batch_size=4,

        validation_split=0.20,

        verbose=0

    )

    prediction = model.predict(

        X_test,

        verbose=0

    )

    prediction = scaler.inverse_transform(

        prediction

    )

    actual = scaler.inverse_transform(

        y_test

    )

    mae = mean_absolute_error(

        actual,

        prediction

    )

    mse = mean_squared_error(

        actual,

        prediction

    )

    rmse = np.sqrt(

        mse

    )

    print("\n", model_name)

    print("MAE :", round(mae,3))

    print("MSE :", round(mse,3))

    print("RMSE :", round(rmse,3))

    return history, prediction, mae, mse, rmse

    # SIMPLE RNN

rnn_history, rnn_prediction, rnn_mae, rnn_mse, rnn_rmse = train_model(

    SimpleRNN,

    "Simple RNN"

)

# LSTM

lstm_history, lstm_prediction, lstm_mae, lstm_mse, lstm_rmse = train_model(

    LSTM,

    "LSTM"

)

# GRU

gru_history, gru_prediction, gru_mae, gru_mse, gru_rmse = train_model(

    GRU,

    "GRU"

)

# MODEL COMPARISON

results = pd.DataFrame({

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

print(results)

# BEST MODEL

best_model = results.sort_values(

    by="RMSE"

).iloc[0]

print("\nBest Time Series Model")

print(best_model)

# ACTUAL VS PREDICTED

actual = scaler.inverse_transform(

    y_test

)

plt.figure(figsize=(12,5))

plt.plot(

    actual,

    label="Actual",

    linewidth=3

)

plt.plot(

    rnn_prediction,

    label="RNN"

)

plt.plot(

    lstm_prediction,

    label="LSTM"

)

plt.plot(

    gru_prediction,

    label="GRU"

)

plt.title("Actual vs Predicted")

plt.legend()

plt.grid(True)

plt.show()

# FINAL RESULTS

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)


# Best Classification Model

best_classification = classification_results.loc[ classification_results["Accuracy"].idxmax() ]

print("\nBest Classification Model")
print(best_classification)


# Best Regression Model

best_regression = regression_results.loc[ regression_results["RMSE"].idxmin() ]

print("\nBest Regression Model")
print(best_regression)


# Best Clustering Result

print("\nBest Clustering Result")
print(f"Optimal Clusters : 3")
print(f"Silhouette Score : {round(score,4)}")


# Best Deep Learning Model

best_deep_learning = results.loc[ results["RMSE"].idxmin() ]

print("\nBest Deep Learning Model")
print(best_deep_learning)

# FINAL CONCLUSION

print("\n" + "="*70)
print("FINAL PROJECT CONCLUSION")
print("="*70)

print("""

This project presented a complete end-to-end Data Analytics,
Machine Learning, and Deep Learning workflow using the
SpaceX Missions Dataset.

The project applied NumPy, Pandas, Seaborn, Feature Engineering,
Correlation Analysis, Classification, Regression, K-Means
Clustering, and Time Series Forecasting using Simple RNN,
LSTM, and GRU.

The models were evaluated using Accuracy, Precision, Recall,
F1 Score, MAE, MSE, RMSE, and R² Score to identify the
best-performing models.

Overall, the project successfully demonstrated data
preprocessing, visualization, predictive modeling,
clustering, and forecasting, achieving all the objectives
of the assignment.

""")

