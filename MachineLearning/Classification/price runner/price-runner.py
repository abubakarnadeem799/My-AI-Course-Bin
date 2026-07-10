# pricerunner_aggregate.csv

# MACHINE LEARNING

# CLASSIFICATION

# IMPORT LIBRARIES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report

)



# LOAD DATASET

df = pd.read_csv(

    "Data_Sets/Machine Learning_Classification/pricerunner_aggregate.csv"

)


# HEAD

print(df.head())


# TAIL

print(df.tail())


# SHAPE

print("\nShape :", df.shape)


# COLUMNS

print("\nColumns")

print(df.columns)


# INFO

print("\nDataset Information")

df.info()


# DATA TYPES

print("\nData Types")

print(df.dtypes)


# DESCRIBE

print("\nStatistical Summary")

print(df.describe())


# MISSING VALUES

print("\nMissing Values")

print(df.isnull().sum())


# DUPLICATE VALUES

print("\nDuplicate Rows :", df.duplicated().sum())


# REMOVE DUPLICATES

df = df.drop_duplicates()


# REMOVE EXTRA SPACES

df.columns = df.columns.str.strip()


# RENAME COLUMNS

df.columns = [

    column.replace(" ", "_")

    for column in df.columns

]


# UPDATED COLUMNS

print("\nUpdated Columns")

print(df.columns)


# TARGET DISTRIBUTION

print("\nCategory Distribution")

print(

    df["Category_Label"].value_counts()

)


# REMOVE MISSING TARGET

df = df.dropna(

    subset=["Category_Label"]

)


# FILL TEXT MISSING VALUES

df["Product_Title"] = df["Product_Title"].fillna("Unknown")

df["Cluster_Label"] = df["Cluster_Label"].fillna("Unknown")


# FILL NUMERIC MISSING VALUES

numeric_columns = [

    "Product_ID",

    "Merchant_ID",

    "Cluster_ID",

    "Category_ID"

]

for column in numeric_columns:

    df[column] = df[column].fillna(

        df[column].median()

    )


# LABEL ENCODE CLUSTER LABEL

cluster_encoder = LabelEncoder()

df["Cluster_Label"] = cluster_encoder.fit_transform(

    df["Cluster_Label"]

)


# LABEL ENCODE TARGET

target_encoder = LabelEncoder()

df["Category_Label"] = target_encoder.fit_transform(

    df["Category_Label"]

)


# TF-IDF VECTORIZATION

tfidf = TfidfVectorizer(

    stop_words="english",

    max_features=500

)

title_features = tfidf.fit_transform(

    df["Product_Title"]

)


# NUMERIC FEATURES

numeric_features = df[

    [

        "Product_ID",

        "Merchant_ID",

        "Cluster_ID",

        "Category_ID",

        "Cluster_Label"

    ]

].values


# COMBINE FEATURES

X = hstack(

    [

        numeric_features,

        title_features

    ]

)


# TARGET

y = df["Category_Label"]


# FEATURE SHAPE

print("\nFeature Matrix Shape :", X.shape)

print("Target Shape :", y.shape)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split( X, y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# TRAIN TEST SHAPES

print("\nTraining Features :", X_train.shape)

print("Testing Features :", X_test.shape)

print("Training Target :", y_train.shape)

print("Testing Target :", y_test.shape)


# DATASET READY

print("\nDataset is Ready for Classification Models")


# CATEGORY DISTRIBUTION

plt.figure(figsize=(12,6))

sns.countplot(

    x=df["Category_Label"]

)

plt.title("Category Distribution")

plt.xlabel("Category")

plt.ylabel("Count")

plt.xticks(rotation=90)

plt.show()


# TOP 20 CATEGORIES

plt.figure(figsize=(14,6))

df["Category_Label"].value_counts().head(20).plot( kind="bar")

plt.title("Top 20 Categories")

plt.xlabel("Category")

plt.ylabel("Count")

plt.show()


# MERCHANT DISTRIBUTION

plt.figure(figsize=(12,6))

sns.countplot(

    x=df["Merchant_ID"]

)

plt.title("Merchant Distribution")

plt.xticks(rotation=90)

plt.show()


# CLUSTER DISTRIBUTION

plt.figure(figsize=(12,6))

sns.countplot(

    x=df["Cluster_ID"]

)

plt.title("Cluster Distribution")

plt.xticks(rotation=90)

plt.show()


print("\nPreprocessing Completed Successfully.")


# TRAIN AND EVALUATE MODEL

def evaluate_model(model, model_name):

    model.fit(

        X_train,

        y_train

    )

    prediction = model.predict(

        X_test

    )

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

    print("\n" + "=" * 70)

    print(model_name)

    print("=" * 70)

    print("Accuracy :", accuracy)

    print("Precision :", precision)

    print("Recall :", recall)

    print("F1 Score :", f1)

    print("\nClassification Report")

    print(

        classification_report(

            y_test,

            prediction,

            zero_division=0

        )

    )

    return (

        model,

        prediction,

        accuracy,

        precision,

        recall,

        f1

    )


# LOGISTIC REGRESSION

logistic_model = LogisticRegression(

    max_iter=1000,

    random_state=42,

    n_jobs=-1

)

(

    logistic_model,

    logistic_prediction,

    logistic_accuracy,

    logistic_precision,

    logistic_recall,

    logistic_f1

) = evaluate_model(

    logistic_model,

    "Logistic Regression"

)


# DECISION TREE

decision_tree_model = DecisionTreeClassifier(

    random_state=42

)

(

    decision_tree_model,

    decision_tree_prediction,

    decision_tree_accuracy,

    decision_tree_precision,

    decision_tree_recall,

    decision_tree_f1

) = evaluate_model(

    decision_tree_model,

    "Decision Tree"

)


# RANDOM FOREST

random_forest_model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

(

    random_forest_model,

    random_forest_prediction,

    random_forest_accuracy,

    random_forest_precision,

    random_forest_recall,

    random_forest_f1

) = evaluate_model(

    random_forest_model,

    "Random Forest"

)


# K NEAREST NEIGHBOR

knn_model = KNeighborsClassifier(

    n_neighbors=5

)

(

    knn_model,

    knn_prediction,

    knn_accuracy,

    knn_precision,

    knn_recall,

    knn_f1

) = evaluate_model(

    knn_model,

    "K-Nearest Neighbor"

)


# SUPPORT VECTOR MACHINE

svm_model = SVC(

    kernel="linear",

    random_state=42

)

(

    svm_model,

    svm_prediction,

    svm_accuracy,

    svm_precision,

    svm_recall,

    svm_f1

) = evaluate_model(

    svm_model,

    "Support Vector Machine"

)


# MODEL COMPARISON

comparison = pd.DataFrame(

    {

        "Model":[

            "Logistic Regression",

            "Decision Tree",

            "Random Forest",

            "K-Nearest Neighbor",

            "Support Vector Machine"

        ],

        "Accuracy":[

            logistic_accuracy,

            decision_tree_accuracy,

            random_forest_accuracy,

            knn_accuracy,

            svm_accuracy

        ],

        "Precision":[

            logistic_precision,

            decision_tree_precision,

            random_forest_precision,

            knn_precision,

            svm_precision

        ],

        "Recall":[

            logistic_recall,

            decision_tree_recall,

            random_forest_recall,

            knn_recall,

            svm_recall

        ],

        "F1 Score":[

            logistic_f1,

            decision_tree_f1,

            random_forest_f1,

            knn_f1,

            svm_f1

        ]

    }

)


# SORT RESULTS

comparison = comparison.sort_values(

    by="Accuracy",

    ascending=False

).reset_index(

    drop=True

)


# MODEL COMPARISON TABLE

print("\n")

print("=" * 70)

print("MODEL COMPARISON")

print("=" * 70)

print(comparison)


# SAVE RESULTS

comparison.to_csv(

    "Classification_Model_Comparison.csv",

    index=False

)


# BEST MODEL

print("\n")

print("=" * 70)

print("BEST MODEL")

print("=" * 70)

print(comparison.head(1))

# MODEL VISUALIZATION

# CONFUSION MATRIX

def plot_confusion_matrix(prediction, model_name):

    cm = confusion_matrix(

        y_test,

        prediction

    )

    plt.figure(figsize=(8,6))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )

    plt.title(model_name)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()


# LOGISTIC REGRESSION CONFUSION MATRIX

plot_confusion_matrix(

    logistic_prediction,

    "Logistic Regression"

)


# DECISION TREE CONFUSION MATRIX

plot_confusion_matrix(

    decision_tree_prediction,

    "Decision Tree"

)


# RANDOM FOREST CONFUSION MATRIX

plot_confusion_matrix(

    random_forest_prediction,

    "Random Forest"

)


# KNN CONFUSION MATRIX

plot_confusion_matrix(

    knn_prediction,

    "K-Nearest Neighbor"

)


# SUPPORT VECTOR MACHINE CONFUSION MATRIX

plot_confusion_matrix(

    svm_prediction,

    "Support Vector Machine"

)


# RANDOM FOREST FEATURE IMPORTANCE

numeric_features = [

    "Product_ID",

    "Merchant_ID",

    "Cluster_ID",

    "Category_ID",

    "Cluster_Label"

]

importance = pd.DataFrame(

    {

        "Feature": numeric_features,

        "Importance": random_forest_model.feature_importances_[:5]

    }

)

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nRandom Forest Feature Importance")

print(importance)


# FEATURE IMPORTANCE PLOT

plt.figure(figsize=(10,6))

sns.barplot(

    data=importance,

    x="Importance",

    y="Feature"

)

plt.title("Random Forest Feature Importance")

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.grid(True)

plt.show()


# ACCURACY COMPARISON

plt.figure(figsize=(10,6))

sns.barplot(

    data=comparison,

    x="Model",

    y="Accuracy"

)

plt.title("Accuracy Comparison")

plt.xticks(rotation=20)

plt.grid(True)

plt.show()


# PRECISION COMPARISON

plt.figure(figsize=(10,6))

sns.barplot(

    data=comparison,

    x="Model",

    y="Precision"

)

plt.title("Precision Comparison")

plt.xticks(rotation=20)

plt.grid(True)

plt.show()


# RECALL COMPARISON

plt.figure(figsize=(10,6))

sns.barplot(

    data=comparison,

    x="Model",

    y="Recall"

)

plt.title("Recall Comparison")

plt.xticks(rotation=20)

plt.grid(True)

plt.show()


# F1 SCORE COMPARISON

plt.figure(figsize=(10,6))

sns.barplot(

    data=comparison,

    x="Model",

    y="F1 Score"

)

plt.title("F1 Score Comparison")

plt.xticks(rotation=20)

plt.grid(True)

plt.show()


# BEST MODEL

best_model = comparison.iloc[0]


print("\n")

print("="*70)

print("BEST CLASSIFICATION MODEL")

print("="*70)

print(best_model)


# FINAL RESULTS

print("\n")

print("="*70)

print("FINAL RESULTS")

print("="*70)

print("Best Model :", best_model["Model"])

print("Accuracy :", round(best_model["Accuracy"],4))

print("Precision :", round(best_model["Precision"],4))

print("Recall :", round(best_model["Recall"],4))

print("F1 Score :", round(best_model["F1 Score"],4))


# SAVE RESULTS

comparison.to_csv(

    "Classification_Model_Comparison.csv",

    index=False

)

print("\nComparison table saved successfully.")


# FINAL CONCLUSION

print("\n")

print("="*70)

