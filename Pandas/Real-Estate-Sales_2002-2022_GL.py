#  Real_Estate_Sales_2001-2022_GL

# PANDAS
# import pandas as pd

import pandas as pd

df = pd.read_csv( "Data_Sets/Data Science_Assignments/Real_Estate_Sales_2001-2022_GL-Short.csv" )


# HEAD

print(df.head())


# TAIL

print(df.tail())


# SHAPE

print(df.shape)


# COLUMNS

print(df.columns)


# INFO

print(df.info())


# DATA TYPES

print(df.dtypes)


# DESCRIBE

print(df.describe())


# MISSING VALUES

print(df.isnull().sum())


# DUPLICATES

print(df.duplicated().sum())


# REMOVE DUPLICATES

df = df.drop_duplicates()


# UNIQUE PROPERTY TYPES

print(df["Property Type"].unique())


# PROPERTY TYPE COUNT

print(df["Property Type"].value_counts())


# UNIQUE TOWNS

print(df["Town"].nunique())


# DATE CONVERSION

df["Date Recorded"] = pd.to_datetime(

    df["Date Recorded"],

    errors="coerce"

)


# SORT BY SALE AMOUNT

print(

    df.sort_values(

        by="Sale Amount",

        ascending=False

    ).head()

)


# GROUP BY TOWN

town_sales = df.groupby(

    "Town"

)["Sale Amount"].mean()

print(town_sales)


# GROUP BY PROPERTY TYPE

property_sales = df.groupby(

    "Property Type"

)["Sale Amount"].mean()

print(property_sales)


# GROUP BY LIST YEAR

year_sales = df.groupby(

    "List Year"

)["Sale Amount"].mean()

print(year_sales)


# HIGH VALUE SALES

high_value = df[

    df["Sale Amount"] > 500000

]

print(high_value.head())


# RESIDENTIAL PROPERTIES

residential = df[

    df["Property Type"] == "Residential"

]

print(residential.head())


# RESET INDEX

df = df.reset_index(

    drop=True

)