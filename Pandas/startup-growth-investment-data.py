# startup-growth-investment-data.csv

# PANDAS

# IMPORT LIBRARY

import pandas as pd


# LOAD DATASET

df = pd.read_csv( "Data_Sets/Data Science_Assignments/startup_growth_investment_data.csv" )


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


# DUPLICATE VALUES

print(df.duplicated().sum())


# REMOVE DUPLICATES

df = df.drop_duplicates()


# UNIQUE INDUSTRIES

print(df["Industry"].unique())


# NUMBER OF UNIQUE INDUSTRIES

print(df["Industry"].nunique())


# INDUSTRY VALUE COUNTS

print(df["Industry"].value_counts())


# UNIQUE COUNTRIES

print(df["Country"].unique())


# NUMBER OF UNIQUE COUNTRIES

print(df["Country"].nunique())


# COUNTRY VALUE COUNTS

print(df["Country"].value_counts())


# SORT BY INVESTMENT AMOUNT

investment_sorted = df.sort_values( by="Investment Amount (USD)",  ascending=False )

print(investment_sorted.head())


# SORT BY VALUATION

valuation_sorted = df.sort_values( by="Valuation (USD)", ascending=False )

print(valuation_sorted.head())


# GROUP BY INDUSTRY

industry_summary = df.groupby( "Industry" )[ [ "Investment Amount (USD)", "Valuation (USD)",  "Growth Rate (%)" ] ].mean()

print(industry_summary)


# GROUP BY COUNTRY

country_summary = df.groupby(  "Country" )[ [  "Investment Amount (USD)", "Valuation (USD)"  ] ].mean()

print(country_summary)


# GROUP BY YEAR FOUNDED

year_summary = df.groupby( "Year Founded" )[  [ "Investment Amount (USD)", "Valuation (USD)" ] ].mean()

print(year_summary)


# HIGHEST INVESTMENT STARTUPS

highest_investment = df.nlargest( 10, "Investment Amount (USD)" )

print(highest_investment)


# HIGHEST VALUATION STARTUPS

highest_valuation = df.nlargest( 10, "Valuation (USD)")

print(highest_valuation)


# HIGH GROWTH STARTUPS

high_growth = df[ df["Growth Rate (%)"] > 100 ]

print(high_growth)


# AI STARTUPS

ai_startups = df[  df["Industry"] == "AI" ]

print(ai_startups)


# FINTECH STARTUPS

fintech_startups = df[  df["Industry"] == "Fintech" ]

print(fintech_startups)


# SAAS STARTUPS

saas_startups = df[ df["Industry"] == "SaaS" ]

print(saas_startups)


# RESET INDEX

df = df.reset_index( drop=True )

print(df.head())