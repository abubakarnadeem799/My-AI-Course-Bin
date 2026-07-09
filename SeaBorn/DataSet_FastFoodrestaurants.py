# FastFoodRestaurants.csv

# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET

df = pd.read_csv("Data_Sets/Data Science_Assignments/FastFoodRestaurants.csv")

# DATA CLEANING

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df = df.drop_duplicates()

df = df.dropna(subset=["latitude", "longitude"])

sns.set_style("whitegrid")

# RESTAURANT COUNT

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    y="name",
    order=df["name"].value_counts().index
)

plt.title("Restaurant Count")
plt.xlabel("Count")
plt.ylabel("Restaurant")
plt.tight_layout()
plt.show()


# STATE COUNT

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    y="province",
    order=df["province"].value_counts().head(15).index
)

plt.title("Top 15 States")
plt.xlabel("Count")
plt.ylabel("State")
plt.tight_layout()
plt.show()


# CITY COUNT

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    y="city",
    order=df["city"].value_counts().head(15).index
)

plt.title("Top 15 Cities")
plt.xlabel("Count")
plt.ylabel("City")
plt.tight_layout()
plt.show()


# LATITUDE DISTRIBUTION

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="latitude",
    kde=True,
    bins=30
)

plt.title("Latitude Distribution")
plt.xlabel("Latitude")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# LONGITUDE DISTRIBUTION

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="longitude",
    kde=True,
    bins=30
)

plt.title("Longitude Distribution")
plt.xlabel("Longitude")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# LATITUDE BOX PLOT

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["latitude"]
)

plt.title("Latitude Box Plot")
plt.tight_layout()
plt.show()


# LONGITUDE BOX PLOT

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["longitude"]
)

plt.title("Longitude Box Plot")
plt.tight_layout()
plt.show()


# LATITUDE VIOLIN PLOT

plt.figure(figsize=(8,5))

sns.violinplot(
    x=df["latitude"]
)

plt.title("Latitude Violin Plot")
plt.tight_layout()
plt.show()


# LONGITUDE VIOLIN PLOT

plt.figure(figsize=(8,5))

sns.violinplot(
    x=df["longitude"]
)

plt.title("Longitude Violin Plot")
plt.tight_layout()
plt.show()


# RESTAURANT LOCATIONS

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="longitude",
    y="latitude",
    alpha=0.7
)

plt.title("Restaurant Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()


# TOP 10 RESTAURANT BRANDS

top_restaurants = df["name"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_restaurants.values,
    y=top_restaurants.index
)

plt.title("Top 10 Restaurant Brands")
plt.xlabel("Count")
plt.ylabel("Restaurant")
plt.tight_layout()
plt.show()


# CORRELATION HEATMAP

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(6,5))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# PAIR PLOT

sns.pairplot(
    numeric_df,
    diag_kind="hist"
)


# ZIP CODE COUNT

if "postalCode" in df.columns:

    plt.figure(figsize=(12,6))

    sns.countplot(
        data=df,
        y="postalCode",
        order=df["postalCode"].value_counts().head(15).index
    )

    plt.title("Top 15 Postal Codes")
    plt.tight_layout()
    plt.show()


# COUNTRY COUNT

if "country" in df.columns:

    plt.figure(figsize=(6,4))

    sns.countplot(
        data=df,
        x="country"
    )

    plt.title("Restaurants by Country")
    plt.tight_layout()
    plt.show()


# RESTAURANTS BY STATE

state_counts = df["province"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=state_counts.values,
    y=state_counts.index
)

plt.title("Top 10 States")
plt.xlabel("Restaurant Count")
plt.ylabel("State")
plt.tight_layout()
plt.show()


# RESTAURANTS BY CITY

city_counts = df["city"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=city_counts.values,
    y=city_counts.index
)

plt.title("Top 10 Cities")
plt.xlabel("Restaurant Count")
plt.ylabel("City")
plt.tight_layout()
plt.show()