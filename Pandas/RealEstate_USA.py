# RealEstate-USA.csv

# IMPORT LIBRARY

import pandas as pd


# PANDAS

# LOAD DATASET

df = pd.read_csv("Data_Sets/Data Science_Assignments/RealEstate-USA.csv")


# DISPLAY FIRST 5 RECORDS

print("\nFirst 5 Records")
print(df.head())


# DISPLAY LAST 5 RECORDS

print("\nLast 5 Records")
print(df.tail())


# DATASET SHAPE

print("\nDataset Shape")
print(df.shape)


# COLUMN NAMES

print("\nColumn Names")
print(df.columns)


# DATASET INFORMATION

print("\nDataset Information")
print(df.info())


# DATA TYPES

print("\nData Types")
print(df.dtypes)


# STATISTICAL SUMMARY

print("\nStatistical Summary")
print(df.describe())


# MISSING VALUES

print("\nMissing Values")
print(df.isnull().sum())


# DUPLICATE RECORDS

print("\nDuplicate Records")
print(df.duplicated().sum())


# REMOVE DUPLICATES

df = df.drop_duplicates()

print("\nShape After Removing Duplicates")
print(df.shape)


# UNIQUE PROPERTY STATUS

print("\nUnique Property Status")
print(df["status"].unique())


# TOTAL UNIQUE PROPERTY STATUS

print("\nNumber of Property Status")
print(df["status"].nunique())


# PROPERTY STATUS COUNT

print("\nProperty Status Count")
print(df["status"].value_counts())


# UNIQUE STATES

print("\nUnique States")
print(df["state"].unique())


# TOTAL STATES

print("\nNumber of States")
print(df["state"].nunique())


# UNIQUE CITIES

print("\nUnique Cities")
print(df["city"].nunique())


# CONVERT DATE COLUMN

df["prev_sold_date"] = pd.to_datetime(

    df["prev_sold_date"],

    errors="coerce"

)

print("\nUpdated Data Types")
print(df.dtypes)


# SORT BY PRICE

print("\nLowest Price Houses")
print(

    df.sort_values(

        by="price"

    ).head()

)


# SORT BY HOUSE SIZE

print("\nLargest Houses")
print(

    df.sort_values(

        by="house_size",

        ascending=False

    ).head()

)


# GROUP BY STATE

print("\nAverage House Price by State")

state_price = df.groupby("state")["price"].mean()

print(state_price)


# GROUP BY PROPERTY STATUS

print("\nAverage House Price by Status")

status_price = df.groupby("status")["price"].mean()

print(status_price)


# GROUP BY CITY

print("\nAverage House Price by City")

city_price = df.groupby("city")["price"].mean()

print(city_price.head(20))


# FILTER EXPENSIVE HOUSES

print("\nHouses Above $1,000,000")

expensive_houses = df[

    df["price"] > 1000000

]

print(expensive_houses.head())


# FILTER SOLD HOUSES

print("\nSold Houses")

sold_houses = df[

    df["status"].str.contains(

        "sold",

        case=False,

        na=False

    )

]

print(sold_houses.head())


# RESET INDEX

df = df.reset_index(

    drop=True

)

print("\nDataset After Reset Index")

print(df.head())