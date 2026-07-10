# Recipe Review and Feedback.csv

# MACHINE LEARNING CLASSIFICATION

# IMPORT LIBRARIES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# MACHINE LEARNING MODELS

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# METRICS

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# LOAD DATASET

df = pd.read_csv(
    "Data_Sets/Machine Learning_Classification/Recipe Reviews and User Feedback Dataset.csv"
)

print(df.head())
print(df.tail())

print("\nShape :", df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Information")
df.info()

print("\nData Types")
print(df.dtypes)

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

# DATA CLEANING

df = df.drop_duplicates()

df.columns = df.columns.str.strip()

df.columns = [
    col.replace(" ", "_")
    for col in df.columns
]

if "Unnamed:_0" in df.columns:
    df.drop(columns="Unnamed:_0", inplace=True)

print("\nStars Distribution")
print(df["stars"].value_counts())

df = df.dropna(subset=["stars"])

df["text"] = df["text"].fillna("No Review")
df["recipe_name"] = df["recipe_name"].fillna("Unknown")
df["user_name"] = df["user_name"].fillna("Unknown")

numeric_columns = [
    "recipe_number",
    "recipe_code",
    "comment_id",
    "user_id",
    "user_reputation",
    "reply_count",
    "thumbs_up",
    "thumbs_down",
    "best_score"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# FEATURE ENGINEERING


df["created_at"] = pd.to_datetime(
    df["created_at"],
    unit="s",
    errors="coerce"
)

df["Review_Year"] = df["created_at"].dt.year
df["Review_Month"] = df["created_at"].dt.month
df["Review_Day"] = df["created_at"].dt.day

df["Review_Year"] = df["Review_Year"].fillna(0)
df["Review_Month"] = df["Review_Month"].fillna(0)
df["Review_Day"] = df["Review_Day"].fillna(0)

# LABEL ENCODING

recipe_encoder = LabelEncoder()
df["recipe_name"] = recipe_encoder.fit_transform(df["recipe_name"])

user_encoder = LabelEncoder()
df["user_name"] = user_encoder.fit_transform(df["user_name"])

target_encoder = LabelEncoder()
df["stars"] = target_encoder.fit_transform(df["stars"])


# TF-IDF

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    min_df=2,
    max_df=0.95
)

text_features = tfidf.fit_transform(df["text"])

scaler = StandardScaler()

numeric_features = scaler.fit_transform(
    df[
        [
            "recipe_number",
            "recipe_code",
            "recipe_name",
            "comment_id",
            "user_id",
            "user_name",
            "user_reputation",
            "reply_count",
            "thumbs_up",
            "thumbs_down",
            "best_score",
            "Review_Year",
            "Review_Month",
            "Review_Day",
        ]
    ]
)

X = hstack(
    [
        numeric_features,
        text_features,
    ]
)

y = df["stars"]

print("\nFeature Matrix Shape :", X.shape)
print("Target Shape :", y.shape)

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Features :", X_train.shape)
print("Testing Features :", X_test.shape)
print("Training Target :", y_train.shape)
print("Testing Target :", y_test.shape)

