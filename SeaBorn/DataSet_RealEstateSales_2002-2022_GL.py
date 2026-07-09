# Real_Estate_Sales_2001-2022_GL

# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET

df = pd.read_csv( "Data_Sets/Data Science_Assignments/Real_Estate_Sales_2001-2022_GL-Short.csv" )

# DATA CLEANING

df["Date Recorded"] = pd.to_datetime(
    df["Date Recorded"],
    errors="coerce"
)

numeric_columns = [
    "Assessed Value",
    "Sale Amount",
    "Sales Ratio",
    "List Year"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.drop_duplicates()

sns.set_style("whitegrid")


# SALE AMOUNT HISTOGRAM

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="Sale Amount",
    bins=30,
    kde=True
)

plt.title("Sale Amount Distribution")
plt.xlabel("Sale Amount")
plt.ylabel("Frequency")
plt.show()


# ASSESSED VALUE HISTOGRAM

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="Assessed Value",
    bins=30,
    kde=True
)

plt.title("Assessed Value Distribution")
plt.xlabel("Assessed Value")
plt.ylabel("Frequency")
plt.show()


# PROPERTY TYPE COUNT

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    x="Property Type"
)

plt.title("Property Type Count")
plt.xticks(rotation=45)
plt.show()


# RESIDENTIAL TYPE COUNT

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    x="Residential Type"
)

plt.title("Residential Type Count")
plt.xticks(rotation=45)
plt.show()


# LIST YEAR COUNT

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    x="List Year"
)

plt.title("Properties by List Year")
plt.xticks(rotation=45)
plt.show()


# SALE AMOUNT BOX PLOT

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["Sale Amount"]
)

plt.title("Sale Amount Box Plot")
plt.show()


# ASSESSED VALUE BOX PLOT

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["Assessed Value"]
)

plt.title("Assessed Value Box Plot")
plt.show()


# SALE AMOUNT VIOLIN PLOT

plt.figure(figsize=(8,5))

sns.violinplot(
    x=df["Sale Amount"]
)

plt.title("Sale Amount Violin Plot")
plt.show()


# ASSESSED VALUE VIOLIN PLOT

plt.figure(figsize=(8,5))

sns.violinplot(
    x=df["Assessed Value"]
)

plt.title("Assessed Value Violin Plot")
plt.show()


# ASSESSED VALUE VS SALE AMOUNT

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="Assessed Value",
    y="Sale Amount"
)

plt.title("Assessed Value vs Sale Amount")
plt.xlabel("Assessed Value")
plt.ylabel("Sale Amount")
plt.show()


# SALES RATIO VS SALE AMOUNT

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="Sales Ratio",
    y="Sale Amount"
)

plt.title("Sales Ratio vs Sale Amount")
plt.xlabel("Sales Ratio")
plt.ylabel("Sale Amount")
plt.show()


# TOP 10 TOWNS

top_towns = df["Town"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_towns.values,
    y=top_towns.index
)

plt.title("Top 10 Towns")
plt.xlabel("Number of Properties")
plt.ylabel("Town")
plt.show()


# AVERAGE SALE AMOUNT BY PROPERTY TYPE

property_avg = (
    df.groupby("Property Type")["Sale Amount"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=property_avg,
    x="Property Type",
    y="Sale Amount"
)

plt.title("Average Sale Amount by Property Type")
plt.xticks(rotation=45)
plt.show()


# AVERAGE SALE AMOUNT BY YEAR

year_avg = (
    df.groupby("List Year")["Sale Amount"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(10,6))

sns.lineplot(
    data=year_avg,
    x="List Year",
    y="Sale Amount",
    marker="o"
)

plt.title("Average Sale Amount by Year")
plt.xlabel("List Year")
plt.ylabel("Average Sale Amount")
plt.show()


# CORRELATION HEATMAP

numeric_df = df[
    [
        "Assessed Value",
        "Sale Amount",
        "Sales Ratio",
        "List Year"
    ]
]

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

plt.show()