# RealEstate-USA.csv

# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET

df = pd.read_csv("Data_Sets/Data Science_Assignments/RealEstate-USA.csv")

# DATA CLEANING

df["prev_sold_date"] = pd.to_datetime(
    df["prev_sold_date"],
    errors="coerce"
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["house_size"] = pd.to_numeric(df["house_size"], errors="coerce")
df["bed"] = pd.to_numeric(df["bed"], errors="coerce")
df["bath"] = pd.to_numeric(df["bath"], errors="coerce")

df = df.drop_duplicates()

sns.set_style("whitegrid")


# PRICE HISTOGRAM

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="price",
    bins=30,
    kde=True
)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


# HOUSE SIZE HISTOGRAM

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="house_size",
    bins=30,
    kde=True
)

plt.title("House Size Distribution")
plt.xlabel("House Size")
plt.ylabel("Frequency")
plt.show()


# PROPERTY STATUS COUNT

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="status"
)

plt.title("Property Status")
plt.xticks(rotation=30)
plt.show()


# BEDROOM COUNT

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="bed"
)

plt.title("Bedrooms")
plt.show()


# BATHROOM COUNT

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="bath"
)

plt.title("Bathrooms")
plt.show()


# PRICE BOX PLOT

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["price"]
)

plt.title("Price Box Plot")
plt.show()


# HOUSE SIZE BOX PLOT

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["house_size"]
)

plt.title("House Size Box Plot")
plt.show()


# PRICE VIOLIN PLOT

plt.figure(figsize=(8,5))

sns.violinplot(
    x=df["price"]
)

plt.title("Price Violin Plot")
plt.show()


# HOUSE SIZE VIOLIN PLOT

plt.figure(figsize=(8,5))

sns.violinplot(
    x=df["house_size"]
)

plt.title("House Size Violin Plot")
plt.show()


# PRICE VS HOUSE SIZE

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="house_size",
    y="price"
)

plt.title("Price vs House Size")
plt.show()


# BEDROOMS VS PRICE

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="bed",
    y="price"
)

plt.title("Bedrooms vs Price")
plt.show()


# TOP 10 STATES

top_states = df["state"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_states.values,
    y=top_states.index
)

plt.title("Top 10 States")
plt.xlabel("Number of Houses")
plt.ylabel("State")
plt.show()


# TOP 10 CITIES

top_cities = df["city"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_cities.values,
    y=top_cities.index
)

plt.title("Top 10 Cities")
plt.xlabel("Number of Houses")
plt.ylabel("City")
plt.show()


# AVERAGE PRICE BY YEAR

year_price = (
    df.dropna(subset=["prev_sold_date"])
      .assign(Year=df["prev_sold_date"].dt.year)
      .groupby("Year")["price"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(12,6))

sns.lineplot(
    data=year_price,
    x="Year",
    y="price",
    marker="o"
)

plt.title("Average Price by Year")
plt.xlabel("Year")
plt.ylabel("Average Price")
plt.show()


# CORRELATION HEATMAP

numeric_df = df[[
    "price",
    "bed",
    "bath",
    "acre_lot",
    "house_size"
]]

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()


# PAIR PLOT

sns.pairplot(
    numeric_df,
    diag_kind="hist"
)

