# SpaceX SPCX Stock Price 30m Interval-spacex.csv

"""
OBJECTIVE:

The objective of this task is to analyze SpaceX stock market data using Python and Machine Learning techniques.

This task performs numerical analysis, exploratory data analysis (EDA), feature engineering, visualization,
classification, regression, clustering and ensemble learning to discover patterns in stock prices and trading volume.

The task uses NumPy for numerical calculations, Pandas for data manipulation, Seaborn for visualization,
and Scikit-learn for predictive modeling.

Finally, multiple Machine Learning models are compared to identify the best-performing algorithm based on
evaluation metrics.

"""

# Import libraries

# NumPy is used for numerical computing, mathematical (calculations, statistical analysis, and array operations).
import numpy as np

# Pandas is used to load, clean, manipulate, and analyze ( structured data stored in tables (DataFrames) ).
import pandas as pd

# Matplotlib is the base visualization library used to create charts, graphs, and plots.
import matplotlib.pyplot as plt

# Seaborn is built on Matplotlib and is used to create attractive statistical visualizations.
import seaborn as sns

# Splits the dataset into training and testing sets for Machine Learning models.
from sklearn.model_selection import train_test_split

# StandardScaler standardizes numerical features
# so that each feature has a mean of 0 and a standard deviation of 1.
from sklearn.preprocessing import StandardScaler

# LabelEncoder converts categorical text values into numerical values that Machine Learning algorithms can understand.
from sklearn.preprocessing import LabelEncoder

# Logistic Regression is used for binary and multi-class classification problems.
from sklearn.linear_model import LogisticRegression

# Decision Tree Classifier creates decision rules to classify data into different categories.
from sklearn.tree import DecisionTreeClassifier

# K-Nearest Neighbors (KNN) classifies data based on the nearest neighboring observations.
from sklearn.neighbors import KNeighborsClassifier

# Random Forest Classifier is an Ensemble Learning algorithm that combines multiple decision trees.
from sklearn.ensemble import RandomForestClassifier

# Linear Regression predicts continuous numerical values.
from sklearn.linear_model import LinearRegression

# Decision Tree Regressor predicts continuous values using a tree-based model.
from sklearn.tree import DecisionTreeRegressor

# Random Forest Regressor combines multiple decision trees to improve prediction accuracy.
from sklearn.ensemble import RandomForestRegressor

# KMeans groups similar observations into clusters without using target labels.
from sklearn.cluster import KMeans

# Accuracy measures the percentage of correct predictions.
from sklearn.metrics import accuracy_score

# Precision measures how many predicted positive values are actually positive.
from sklearn.metrics import precision_score

# Recall measures how many actual positive values were correctly predicted.
from sklearn.metrics import recall_score

# F1 Score is the harmonic mean of Precision and Recall.
from sklearn.metrics import f1_score

# Confusion Matrix summarizes classification performance.
from sklearn.metrics import confusion_matrix

# Classification Report provides Accuracy, Precision, Recall, and F1 Score together.
from sklearn.metrics import classification_report

# R² Score evaluates Regression model performance.
from sklearn.metrics import r2_score

# Mean Absolute Error (MAE) measures average prediction error.
from sklearn.metrics import mean_absolute_error

# Mean Squared Error (MSE) measures the average squared error.
from sklearn.metrics import mean_squared_error

# Silhouette Score evaluates the quality of clusters created by the KMeans algorithm.
from sklearn.metrics import silhouette_score

# Set a professional visualization style for all charts.
sns.set_style("whitegrid")

# NUMPY

# LOAD DATASET USING NP.GENFROMTXT()
# Load the CSV file using NumPy.
# delimiter="," specifies that the file is comma-separated.
# names=True reads the first row as column names.
# dtype=None automatically detects the data type of each column.
# encoding="utf-8" correctly reads text values.

data = np.genfromtxt("Data_Sets/SpaceX SPCX Stock Price 30m Interval/spacex.csv", delimiter=",", names=True, dtype=None, encoding="utf-8" )



# Display the first five records.
print("\nFirst Five Records")
print(data[:5])

# Display dataset shape.
print("\nDataset Shape")
print(data.shape)

# Display column names.
print("\nColumn Names")
print(data.dtype.names)

# Store each numerical column in a separate variable.

open_price = data["Open"]
high_price = data["High"]
low_price = data["Low"]
close_price = data["Close"]
volume = data["Volume"]

# MEAN
# Calculate average closing price.

print("\nMean Close Price")
print(np.mean(close_price))

# MEDIAN
# Calculate median closing price.

print("\nMedian Close Price")
print(np.median(close_price))

# MAXIMUM
# Find highest closing price.

print("\nMaximum Close Price")
print(np.max(close_price))

# MINIMUM
# Find lowest closing price.

print("\nMinimum Close Price")
print(np.min(close_price))

# STANDARD DEVIATION
# Measure price volatility.

print("\nStandard Deviation")
print(np.std(close_price))

# VARIANCE
# Measure variation in prices.

print("\nVariance")
print(np.var(close_price))

# SUM
# Calculate total trading volume.

print("\nTotal Volume")
print(np.sum(volume))

# AVERAGE
# Calculate average trading volume.

print("\nAverage Volume")
print(np.average(volume))

# PERCENTILES
# Display important percentiles.

print("\n25th Percentile")
print(np.percentile(close_price, 25))

print("\n50th Percentile")
print(np.percentile(close_price, 50))

print("\n75th Percentile")
print(np.percentile(close_price, 75))

# RANGE
# Calculate price range.

print("\nPrice Range")
print(np.ptp(close_price))

# ARGMAX
# Find index of highest closing price.

print("\nIndex of Maximum Close Price")
print(np.argmax(close_price))

# ARGMIN
# Find index of lowest closing price.

print("\nIndex of Minimum Close Price")
print(np.argmin(close_price))

# SORTING
# Sort closing prices.

print("\nSorted Close Prices")
print(np.sort(close_price))

# UNIQUE VALUES
# Display unique closing prices.

print("\nUnique Close Prices")
print(np.unique(close_price))

# CORRELATION
# Calculate correlation between Open and Close prices.

print("\nCorrelation (Open vs Close)")
print( np.corrcoef( open_price, close_price ) )


# COVARIANCE
# Calculate covariance between Open and Close prices.

print("\nCovariance (Open vs Close)")
print( np.cov( open_price, close_price ) )

# CUMULATIVE SUM
# Calculate cumulative trading volume.

print("\nCumulative Volume")
print(np.cumsum(volume))

# DIFFERENCE
# Calculate difference between consecutive closing prices.

print("\nDifference Between Consecutive Close Prices")
print(np.diff(close_price))

# LOGARITHM
# Calculate natural logarithm of closing prices.

print("\nLog of Close Prices")
print(np.log(close_price))

# SQUARE ROOT
# Calculate square root of closing prices.

print("\nSquare Root of Close Prices")
print(np.sqrt(close_price))

# MINIMUM & MAXIMUM USING NUMPY

print("\nMinimum and Maximum Close Price")
print(np.min(close_price), np.max(close_price))

# Pandas
# Load the CSV file into a Pandas DataFrame.
df = pd.read_csv( "Data_Sets/SpaceX SPCX Stock Price 30m Interval/spacex.csv" )

# Display the first 5 rows.
print("\nFirst Five Records")
print(df.head())

# Display the last 5 rows.
print("\nLast Five Records")
print(df.tail())

# Display dataset dimensions (rows, columns).
print("\nDataset Shape")
print(df.shape)

# Display column names.
print("\nColumn Names")
print(df.columns)

# Display data types.
print("\nData Types")
print(df.dtypes)

# Display dataset information.
print("\nDataset Information")
df.info()

# Display descriptive statistics for numerical columns.
print("\nSummary Statistics")
print(df.describe())


# DATA CLEANING
# Count missing values in each column.
print("\nMissing Values")
print(df.isnull().sum())

# Count duplicate rows.
print("\nDuplicate Rows")
print(df.duplicated().sum())

# Remove duplicate rows if any exist.
df.drop_duplicates(inplace=True)

# DATETIME CONVERSION
# Convert Datetime column into datetime format.
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Display updated data types.
print("\nUpdated Data Types")
print(df.dtypes)


# SORTING
# Sort dataset by Close price (Highest to Lowest).
print("\nHighest Closing Prices")
print( df.sort_values( by="Close", ascending=False ).head() )


# FILTERING
# Display records where Close price is greater than Open price.
print("\nBullish Trading Days")
print( df[ df["Close"] > df["Open"] ] )

# Display records where trading volume is above average.
print("\nHigh Volume Trading Days")
print( df[ df["Volume"] > df["Volume"].mean() ] )


# COLUMN SELECTION
# Display selected columns only.
print("\nSelected Columns")
print(df[
        [
            "Datetime",
            "Open",
            "Close",
            "Volume"
        ]
    ]
)


# UNIQUE VALUES
# Count unique values in each column.
print("\nUnique Values")
print(df.nunique())

# CORRELATION MATRIX
# Display correlation between numerical columns.
print("\nCorrelation Matrix")
print( df.corr( numeric_only=True ) )

# MAXIMUM VALUES
# Display row having maximum Close price.
print("\nHighest Closing Price")
print( df.loc[ df["Close"].idxmax() ] )


# MINIMUM VALUES
# Display row having minimum Close price.
print("\nLowest Closing Price")
print( df.loc[ df["Close"].idxmin() ] )

# AGGREGATE FUNCTIONS
# Total trading volume.
print("\nTotal Volume")
print(df["Volume"].sum())

# Average Close price.
print("\nAverage Close Price")
print(df["Close"].mean())

# Highest Close price.
print("\nMaximum Close Price")
print(df["Close"].max())

# Lowest Close price.
print("\nMinimum Close Price")
print(df["Close"].min())

# Median Close price.
print("\nMedian Close Price")
print(df["Close"].median())

# Standard deviation.
print("\nStandard Deviation")
print(df["Close"].std())

# Variance.
print("\nVariance")
print(df["Close"].var())


# Extract Year.
df["Year"] = df["Datetime"].dt.year

# Extract Month.
df["Month"] = df["Datetime"].dt.month

# Extract Day.
df["Day"] = df["Datetime"].dt.day

print("\nDate Features")
print(
    df[
        [
            "Datetime",
            "Year",
            "Month",
            "Day"
        ]
    ].head()
)


# GROUPBY ANALYSIS
# Average closing price by month.
print("\nAverage Close Price by Month")
print( df.groupby("Month")["Close"].mean() )

# Total trading volume by month.
print("\nTotal Volume by Month")
print( df.groupby("Month")["Volume"].sum() )

# Highest Close price by month.
print("\nMaximum Close by Month")
print( df.groupby("Month")["Close"].max() )

# Display the first five rows after all transformations.
print("\nFinal Dataset Preview")
print(df.head())


# FEATURE ENGINEERING

# Calculate the daily price change.
# Positive value = Stock closed higher than it opened.
# Negative value = Stock closed lower than it opened.

df["Price_Change"] = ( df["Close"] - df["Open"] )

print("\nPrice Change")

print(
    df[
        [
            "Open",
            "Close",
            "Price_Change"
        ]
    ].head()
)

# DAILY PRICE RANGE
# Calculate the difference between the highest and lowest trading prices.

df["Price_Range"] = ( df["High"] - df["Low"] )

print("\nDaily Price Range")

print(
    df[
        [
            "High",
            "Low",
            "Price_Range"
        ]
    ].head()
)


# DAILY RETURN (%)
# Calculate the daily percentage return.

df["Return_%"] = ( ( df["Close"] - df["Open"] ) / df["Open"] ) * 100
print("\nDaily Return (%)")

print(
    df[
        [
            "Open",
            "Close",
            "Return_%"
        ]
    ].head()
)


# HIGH-LOW RATIO
# Calculate the ratio between High and Low prices.
# This indicates daily market volatility.

df["High_Low_Ratio"] = ( df["High"] / df["Low"] )
print("\nHigh-Low Ratio")

print(
    df[
        [
            "High",
            "Low",
            "High_Low_Ratio"
        ]
    ].head()
)


# OPEN-CLOSE RATIO
# Calculate the ratio between Open and Close prices.

df["Open_Close_Ratio"] = ( df["Open"] / df["Close"] )
print("\nOpen-Close Ratio")

print(
    df[
        [
            "Open",
            "Close",
            "Open_Close_Ratio"
        ]
    ].head()
)

# 5-DAY MOVING AVERAGE
# Calculate the average Close price over the previous 5 trading days.

df["MA_5"] = ( df["Close"].rolling(window=5).mean() )

print("\n5-Day Moving Average")

print(
    df[
        [
            "Close",
            "MA_5"
        ]
    ].head(10)
)


# 10-DAY MOVING AVERAGE
# Calculate the average Close price over the previous 10 trading days.

df["MA_10"] = ( df["Close"].rolling(window=10).mean() )

print("\n10-Day Moving Average")

print( df[ [ "Close","MA_10" ] ].head(12) )

# 5-DAY VOLATILITY
# Calculate rolling standard deviation to measure price volatility.

df["Volatility"] = ( df["Close"].rolling(window=5).std() )
print("\nRolling Volatility")

print( df[["Close", "Volatility"]].head(10) )


# DAILY PRICE DIRECTION
# Create a target column for Classification.
# 1 = Bullish Day (Close > Open)
# 0 = Bearish Day (Close <= Open)

df["Target"] = ( df["Close"] > df["Open"] ).astype(int)

print("\nTarget Variable")

print(df[
        [
            "Open",
            "Close",
            "Target"
        ]
    ].head()
)
\
# HANDLE MISSING VALUES CREATED BY ROLLING()
# Rolling calculations create NaN values for the first few rows because there are not enough previous observations.

df.bfill(inplace=True)

print("\nMissing Values After Feature Engineering")

print( df.isnull().sum() )


# CORRELATION HEATMAP
# Display the correlation between all numerical columns.

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.show()


# CLOSE PRICE DISTRIBUTION
# Display the distribution of closing prices.

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Close"],
    bins=10,
    kde=True
)

plt.title("Distribution of Closing Price")

plt.xlabel("Close Price")

plt.ylabel("Frequency")

plt.show()


# VOLUME DISTRIBUTION
# Display the distribution of trading volume.

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Volume"],
    bins=10,
    kde=True
)

plt.title("Trading Volume Distribution")

plt.xlabel("Volume")

plt.ylabel("Frequency")

plt.show()


# OPEN VS CLOSE
# Analyze the relationship between Open and Close prices.

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Open",
    y="Close"
)

plt.title("Open Price vs Close Price")

plt.show()


# HIGH VS LOW
# Analyze the relationship between High and Low prices.

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="High",
    y="Low"
)

plt.title("High Price vs Low Price")

plt.show()


# CLOSE PRICE TREND
# Display the closing price over time.

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=df,
    x="Datetime",
    y="Close"
)

plt.title("Closing Price Trend")

plt.xticks(rotation=45)

plt.show()


# VOLUME TREND
# Display trading volume over time.

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=df,
    x="Datetime",
    y="Volume"
)

plt.title("Trading Volume Trend")

plt.xticks(rotation=45)

plt.show()


# BOXPLOT OF CLOSE PRICE
# Detect outliers in closing prices.

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Close"]
)

plt.title("Boxplot of Closing Price")

plt.show()


# PAIRPLOT
# Compare relationships among numerical variables.

sns.pairplot(
    df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]
)

plt.show()


# REGRESSION PLOT
# Display the regression line between Open and Close prices.

plt.figure(figsize=(8, 6))

sns.regplot(
    data=df,
    x="Open",
    y="Close"
)

plt.title("Regression Plot: Open vs Close")

plt.show()



# CLASSIFICATION
# SELECT FEATURES
# Select input features (independent variables)

X = df[
    [
        "Open",
        "High",
        "Low",
        "Volume",
        "Price_Change",
        "Price_Range",
        "Return_%",
        "High_Low_Ratio",
        "Open_Close_Ratio",
        "MA_5",
        "MA_10",
        "Volatility"
    ]
]

# Select target variable

y = df["Target"]

print("\nFeatures")

print(X.head())

print("\nTarget")

print(y.head())

# TRAIN TEST SPLIT


# Split dataset into training and testing data.
# 80% is used for training and 20% for testing.

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=42, stratify=y )

print("\nTraining Shape")

print(X_train.shape)

print("\nTesting Shape")

print(X_test.shape)


# FEATURE SCALING
# Scale numerical features for algorithms
# like Logistic Regression and KNN.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform( X_train )

X_test_scaled = scaler.transform(X_test)


# LOGISTIC REGRESSION
# Create Logistic Regression model.

lr_model = LogisticRegression()

# Train the model.

lr_model.fit( X_train_scaled, y_train )

# Predict testing data.

lr_predictions = lr_model.predict( X_test_scaled )

# Calculate evaluation metrics.

lr_accuracy = accuracy_score( y_test, lr_predictions )

lr_precision = precision_score( y_test, lr_predictions )

lr_recall = recall_score( y_test, lr_predictions )

lr_f1 = f1_score( y_test, lr_predictions )

print("\n Logistic Regression ")

print("Accuracy :", lr_accuracy)

print("Precision :", lr_precision)

print("Recall :", lr_recall)

print("F1 Score :", lr_f1)

print("\nConfusion Matrix")

print( confusion_matrix( y_test, lr_predictions ))

print("\nClassification Report")

print( classification_report( y_test, lr_predictions ) )


# DECISION TREE

dt_model = DecisionTreeClassifier( random_state=42 )

dt_model.fit( X_train, y_train )

dt_predictions = dt_model.predict( X_test )

dt_accuracy = accuracy_score( y_test, dt_predictions )

dt_precision = precision_score( y_test, dt_predictions )

dt_recall = recall_score( y_test, dt_predictions )

dt_f1 = f1_score( y_test, dt_predictions )

print("\n Decision Tree ")

print("Accuracy :", dt_accuracy)

print("Precision :", dt_precision)

print("Recall :", dt_recall)

print("F1 Score :", dt_f1)

print(confusion_matrix( y_test, dt_predictions ))

print(classification_report( y_test, dt_predictions ))



# KNN CLASSIFIER


knn_model = KNeighborsClassifier( n_neighbors=5 )

knn_model.fit( X_train_scaled, y_train )

knn_predictions = knn_model.predict( X_test_scaled )

knn_accuracy = accuracy_score( y_test, knn_predictions )

knn_precision = precision_score( y_test, knn_predictions)

knn_recall = recall_score( y_test, knn_predictions )

knn_f1 = f1_score( y_test, knn_predictions )

print("\n KNN ")

print("Accuracy :", knn_accuracy)

print("Precision :", knn_precision)

print("Recall :", knn_recall)

print("F1 Score :", knn_f1)

print(confusion_matrix( y_test, knn_predictions))

print(classification_report( y_test, knn_predictions ))


# RANDOM FOREST CLASSIFIER

rf_model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)

rf_model.fit( X_train, y_train )

rf_predictions = rf_model.predict( X_test )

rf_accuracy = accuracy_score( y_test, rf_predictions )

rf_precision = precision_score( y_test, rf_predictions )

rf_recall = recall_score( y_test, rf_predictions )

rf_f1 = f1_score( y_test, rf_predictions )

print("\n Random Forest")

print("Accuracy :", rf_accuracy)

print("Precision :", rf_precision)

print("Recall :", rf_recall)

print("F1 Score :", rf_f1)

print(confusion_matrix( y_test, rf_predictions ))

print(classification_report( y_test, rf_predictions ))


# MODEL COMPARISON
classification_results = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Random Forest"
    ],

    "Accuracy":[
        lr_accuracy,
        dt_accuracy,
        knn_accuracy,
        rf_accuracy
    ],

    "Precision":[
        lr_precision,
        dt_precision,
        knn_precision,
        rf_precision
    ],

    "Recall":[
        lr_recall,
        dt_recall,
        knn_recall,
        rf_recall
    ],

    "F1 Score":[
        lr_f1,
        dt_f1,
        knn_f1,
        rf_f1
    ]
})

print("\nClassification Model Comparison")

print(classification_results)

# BEST MODEL

best_model = classification_results.loc[ classification_results["Accuracy"].idxmax()]

print("\nBest Classification Model")

print(best_model)


# REGRESSION

X = df[
    [
        "Open",
        "High",
        "Low",
        "Volume",
        "Price_Change",
        "Price_Range",
        "Return_%",
        "High_Low_Ratio",
        "Open_Close_Ratio",
        "MA_5",
        "MA_10",
        "Volatility"
    ]
]

# Select the target variable (dependent variable).

y = df["Close"]

print("\nFeatures")

print(X.head())

print("\nTarget")

print(y.head())


# TRAIN TEST SPLIT
# Split the dataset into training and testing data.

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.20, random_state=42 )

print("\nTraining Data Shape")

print(X_train.shape)

print("\nTesting Data Shape")

print(X_test.shape)


# FEATURE SCALING
# Standardize the feature values.
# Scaling helps algorithms like Linear Regression.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# LINEAR REGRESSION
# Create Linear Regression model.

lr_model = LinearRegression()

# Train the model.

lr_model.fit( X_train_scaled, y_train )

# Predict Closing Prices.

lr_predictions = lr_model.predict( X_test_scaled )

# Calculate evaluation metrics.

lr_r2 = r2_score( y_test, lr_predictions )

lr_mae = mean_absolute_error( y_test, lr_predictions )

lr_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lr_predictions
    )
)

print("\n Linear Regression ")

print("R² Score :", lr_r2)

print("MAE :", lr_mae)

print("RMSE :", lr_rmse)



# DECISION TREE REGRESSOR
# Create Decision Tree Regression model.

dt_model = DecisionTreeRegressor( random_state=42 )

# Train the model.

dt_model.fit( X_train, y_train )

# Predict Closing Prices.

dt_predictions = dt_model.predict( X_test )

# Evaluate the model.

dt_r2 = r2_score( y_test, dt_predictions )

dt_mae = mean_absolute_error( y_test, dt_predictions )

dt_rmse = np.sqrt( mean_squared_error( y_test, dt_predictions ) )

print("\n Decision Tree Regressor ")

print("R² Score :", dt_r2)

print("MAE :", dt_mae)

print("RMSE :", dt_rmse)


# RANDOM FOREST REGRESSOR   
# Create Random Forest Regression model.

rf_model = RandomForestRegressor(

    n_estimators=100,

    random_state=42
)

# Train the model.

rf_model.fit( X_train, y_train )

# Predict Closing Prices.

rf_predictions = rf_model.predict( X_test )

# Evaluate the model.

rf_r2 = r2_score( y_test, rf_predictions )

rf_mae = mean_absolute_error( y_test, rf_predictions )

rf_rmse = np.sqrt( mean_squared_error( y_test, rf_predictions ) )

print("\n Random Forest Regressor")

print("R² Score :", rf_r2)

print("MAE :", rf_mae)

print("RMSE :", rf_rmse)



# REGRESSION MODEL COMPARISON
# Create a comparison table for all regression models.

regression_results = pd.DataFrame({

    "Model":[

        "Linear Regression",

        "Decision Tree",

        "Random Forest"

    ],

    "R² Score":[ lr_r2, dt_r2, rf_r2 ],

    "MAE":[ lr_mae,dt_mae,rf_mae ],

    "RMSE":[ lr_rmse, dt_rmse, rf_rmse ]
})

print("\nRegression Model Comparison")

print(regression_results)

# BEST REGRESSION MODEL
# Find the model with the highest R² Score.

best_regression = regression_results.loc[regression_results["R² Score"].idxmax()]

print("\nBest Regression Model")

print(best_regression)

# ACTUAL VS PREDICTED VALUES
# Compare actual and predicted values from the best model
# (Random Forest is used here as an example).

comparison = pd.DataFrame({

    "Actual Close Price": y_test.values,

    "Predicted Close Price": rf_predictions

})

print("\nActual vs Predicted Values")

print(comparison.head(10))

# SECTION 9 : CLUSTERING (KMEANS)
# SELECT FEATURES FOR CLUSTERING


cluster_data = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]
]

print("\nClustering Features")

print(cluster_data.head())


# FEATURE SCALING

scaler = StandardScaler()

cluster_scaled = scaler.fit_transform( cluster_data )

print("\nScaled Data Shape")

print(cluster_scaled.shape)


# ELBOW METHOD
# Elbow Method helps determine the
# optimal number of clusters.

wcss = []

# Test cluster values from 1 to 10.

for i in range(1, 11):

    kmeans = KMeans(

        n_clusters=i,

        random_state=42,

        n_init=10
    )

    kmeans.fit(cluster_scaled)

    # Store Within Cluster Sum of Squares.

    wcss.append( kmeans.inertia_ )

# Plot Elbow Curve.

plt.figure(figsize=(8, 5))

plt.plot( range(1, 11),wcss,marker="o")

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.show()

# OPTIMAL K
# Based on the Elbow Method,
# select the optimal number of clusters.

optimal_k = 3

print( "\nSelected Number of Clusters:", optimal_k )

# APPLY KMEANS
# Create KMeans model.

kmeans = KMeans(

    n_clusters=optimal_k,

    random_state=42,

    n_init=10
)

# Fit model and predict clusters.

clusters = kmeans.fit_predict( cluster_scaled )

# Add cluster labels to dataset.

df["Cluster"] = clusters

print("\nCluster Labels Added")

print(
    df[
        [
            "Close",
            "Volume",
            "Cluster"
        ]
    ].head()
)

# SILHOUETTE SCORE
# Measure clustering quality.
# Higher value = Better clustering.

sil_score = silhouette_score( cluster_scaled, clusters )

print( "\nSilhouette Score:", sil_score )


# CLUSTER VISUALIZATION
# Visualize clusters using Close Price and Volume.

plt.figure(figsize=(10, 6))

sns.scatterplot(

    data=df,

    x="Close",

    y="Volume",

    hue="Cluster",

    palette="Set1",

    s=100
)

plt.title( "KMeans Cluster Visualization" )

plt.show()

# CLUSTER ANALYSIS
# Analyze average values within each cluster.

cluster_summary = df.groupby(
    "Cluster"
)[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]
].mean()

print("\nCluster Summary")

print(cluster_summary)


# CLUSTER DISTRIBUTION
# Count observations in each cluster.

print("\nCluster Distribution")

print( df["Cluster"].value_counts() )



# SECTION 10 : ENSEMBLE LEARNING
# Display the performance of the Random Forest model.

print("\nRandom Forest Classification Accuracy")

print(rf_accuracy)

# Display feature importance.

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf_model.feature_importances_

})

# Sort features by importance.

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")

print(feature_importance)




