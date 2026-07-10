# gait.py

# MACHINE LEARNING

# CLASSIFICATION

# IMPORT LIBRARIES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

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

df = pd.read_csv( "Data_Sets/Machine Learning_Classification/gait.csv" )


# HEAD

print(df.head())


# TAIL

print(df.tail())


# SHAPE

print("Shape :", df.shape)


# COLUMNS

print(df.columns)


# INFO

df.info()


# DATA TYPES

print(df.dtypes)


# DESCRIBE

print(df.describe())


# MISSING VALUES

print(df.isnull().sum())


# DUPLICATE VALUES

print("Duplicate Rows :", df.duplicated().sum())


# REMOVE DUPLICATES

df = df.drop_duplicates()


# REMOVE EXTRA SPACES

df.columns = df.columns.str.strip()


# TARGET CLASSES

print(df["condition"].value_counts())


# LABEL ENCODING

encoder = LabelEncoder()

df["condition"] = encoder.fit_transform( df["condition"] )


# FEATURES

X = df.drop( columns="condition" )


# TARGET

y = df["condition"]


# FEATURE SHAPE

print(X.shape)

print(y.shape)


# CHECK CATEGORICAL COLUMNS

categorical_columns = X.select_dtypes( include="object" ).columns

print(categorical_columns)


# ENCODE CATEGORICAL FEATURES

feature_encoder = LabelEncoder()

for column in categorical_columns:  X[column] = feature_encoder.fit_transform( X[column].astype(str) )


# CHECK DATA TYPES

print(X.dtypes)


# MISSING VALUES

print(X.isnull().sum())


# FILL MISSING VALUES

for column in X.columns:

    X[column] = X[column].fillna( X[column].median() )


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split( X, y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# TRAINING SHAPE

print("X Train :", X_train.shape)

print("X Test :", X_test.shape)

print("y Train :", y_train.shape)

print("y Test :", y_test.shape)


# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform( X_train )

X_test = scaler.transform( X_test)


# SCALING COMPLETED

print("Scaling Completed")


# TARGET DISTRIBUTION

plt.figure(figsize=(8,5))

sns.countplot(

    x=y

)

plt.title("Condition Distribution")

plt.xlabel("Condition")

plt.ylabel("Count")

plt.show()


# CORRELATION HEATMAP

plt.figure(figsize=(10,8))

sns.heatmap(

    df.corr(

        numeric_only=True

    ),

    annot=True,

    cmap="coolwarm"

)

plt.title("Correlation Heatmap")

plt.show()


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

        average="weighted"

    )

    recall = recall_score(

        y_test,

        prediction,

        average="weighted"

    )

    f1 = f1_score(

        y_test,

        prediction,

        average="weighted"

    )

    print("\n" + "="*70)

    print(model_name)

    print("="*70)

    print("Accuracy :", accuracy)

    print("Precision :", precision)

    print("Recall :", recall)

    print("F1 Score :", f1)

    print("\nClassification Report")

    print(

        classification_report(

            y_test,

            prediction

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

    random_state=42

)

(

    logistic_model,

    logistic_prediction,

    logistic_accuracy,

    logistic_precision,

    logistic_recall,

    logistic_f1

) = evaluate_model( logistic_model, "Logistic Regression" )


# DECISION TREE

decision_tree_model = DecisionTreeClassifier( random_state=42 )

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

knn_model = KNeighborsClassifier( n_neighbors=5 )

(

    knn_model,

    knn_prediction,

    knn_accuracy,

    knn_precision,

    knn_recall,

    knn_f1

) = evaluate_model( knn_model, "K-Nearest Neighbor" )


# SUPPORT VECTOR MACHINE

svm_model = SVC( kernel="rbf", random_state=42 )

(

    svm_model,

    svm_prediction,

    svm_accuracy,

    svm_precision,

    svm_recall,

    svm_f1

) = evaluate_model( svm_model,"Support Vector Machine" )


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


# MODEL COMPARISON TABLE

print("\n")

print("="*70)

print("MODEL COMPARISON")

print("="*70)

print(comparison)


# SORT MODELS

comparison = comparison.sort_values(  by="Accuracy",  ascending=False )


# BEST MODEL

print("\n")

print("="*70)

print("BEST MODEL")

print("="*70)

print(comparison.head(1))


# MODEL VISUALIZATION

# CONFUSION MATRIX

def plot_confusion_matrix(prediction, model_name):

    cm = confusion_matrix( y_test, prediction )

    plt.figure(figsize=(6,5))

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


# FEATURE NAMES

feature_names = X.columns


# DECISION TREE FEATURE IMPORTANCE

decision_importance = pd.DataFrame(

    {

        "Feature": feature_names,

        "Importance": decision_tree_model.feature_importances_

    }

)

decision_importance = decision_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nDecision Tree Feature Importance")

print(decision_importance)


plt.figure(figsize=(12,6))

sns.barplot(

    data=decision_importance,

    x="Importance",

    y="Feature"

)

plt.title("Decision Tree Feature Importance")

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.show()


# RANDOM FOREST FEATURE IMPORTANCE

forest_importance = pd.DataFrame(

    {

        "Feature": feature_names,

        "Importance": random_forest_model.feature_importances_

    }

)

forest_importance = forest_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nRandom Forest Feature Importance")

print(forest_importance)


plt.figure(figsize=(12,6))

sns.barplot(

    data=forest_importance,

    x="Importance",

    y="Feature"

)

plt.title("Random Forest Feature Importance")

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.show()


# ACCURACY COMPARISON

plt.figure(figsize=(10,6))

sns.barplot(

    data=comparison,

    x="Model",

    y="Accuracy"

)

plt.title("Accuracy Comparison")

plt.xticks(rotation=15)

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

plt.xticks(rotation=15)

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

plt.xticks(rotation=15)

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

plt.xticks(rotation=15)

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

print("\nModel comparison saved successfully.")


