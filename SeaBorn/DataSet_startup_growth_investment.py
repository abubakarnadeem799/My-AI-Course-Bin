# startup_growth_investment_data.csv

# SEABORN

# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# LOAD DATASET

df = pd.read_csv( "Data_Sets/Data Science_Assignments/startup_growth_investment_data.csv" )


# REMOVE DUPLICATES

df = df.drop_duplicates()


# CONVERT NUMERIC COLUMNS

numeric_columns = [

    "Funding Rounds",

    "Investment Amount (USD)",

    "Valuation (USD)",

    "Number of Investors",

    "Year Founded",

    "Growth Rate (%)"

]

for column in numeric_columns:  df[column] = pd.to_numeric( df[column], errors="coerce" )


# REMOVE MISSING VALUES

df = df.dropna(subset=numeric_columns)


# SEABORN STYLE

sns.set_style("whitegrid")


# INVESTMENT AMOUNT HISTOGRAM

plt.figure(figsize=(10,6))

sns.histplot(

    data=df,

    x="Investment Amount (USD)",

    kde=True,

    bins=30

)

plt.title("Investment Amount Distribution")

plt.show()


# VALUATION HISTOGRAM

plt.figure(figsize=(10,6))

sns.histplot(

    data=df,

    x="Valuation (USD)",

    kde=True,

    bins=30

)

plt.title("Valuation Distribution")

plt.show()


# GROWTH RATE HISTOGRAM

plt.figure(figsize=(10,6))

sns.histplot(

    data=df,

    x="Growth Rate (%)",

    kde=True,

    bins=30

)

plt.title("Growth Rate Distribution")

plt.show()


# INDUSTRY COUNT PLOT

plt.figure(figsize=(12,6))

sns.countplot(

    data=df,

    y="Industry",

    order=df["Industry"].value_counts().index

)

plt.title("Industry Count")

plt.show()


# COUNTRY COUNT PLOT

plt.figure(figsize=(12,6))

sns.countplot(

    data=df,

    y="Country",

    order=df["Country"].value_counts().head(10).index

)

plt.title("Top Countries")

plt.show()


# FUNDING ROUNDS COUNT PLOT

plt.figure(figsize=(10,6))

sns.countplot(

    data=df,

    x="Funding Rounds"

)

plt.title("Funding Rounds Count")

plt.show()


# INVESTMENT AMOUNT BOX PLOT

plt.figure(figsize=(10,5))

sns.boxplot(

    x=df["Investment Amount (USD)"]

)

plt.title("Investment Amount Box Plot")

plt.show()


# VALUATION BOX PLOT

plt.figure(figsize=(10,5))

sns.boxplot(

    x=df["Valuation (USD)"]

)

plt.title("Valuation Box Plot")

plt.show()


# INVESTMENT AMOUNT VIOLIN PLOT

plt.figure(figsize=(10,5))

sns.violinplot(

    x=df["Investment Amount (USD)"]

)

plt.title("Investment Amount Violin Plot")

plt.show()


# GROWTH RATE VIOLIN PLOT

plt.figure(figsize=(10,5))

sns.violinplot(

    x=df["Growth Rate (%)"]

)

plt.title("Growth Rate Violin Plot")

plt.show()


# INVESTMENT VS VALUATION

plt.figure(figsize=(10,6))

sns.scatterplot(

    data=df,

    x="Investment Amount (USD)",

    y="Valuation (USD)"

)

plt.title("Investment vs Valuation")

plt.show()


# FUNDING ROUNDS VS GROWTH RATE

plt.figure(figsize=(10,6))

sns.scatterplot(

    data=df,

    x="Funding Rounds",

    y="Growth Rate (%)"

)

plt.title("Funding Rounds vs Growth Rate")

plt.show()


# INVESTORS VS VALUATION

plt.figure(figsize=(10,6))

sns.scatterplot(

    data=df,

    x="Number of Investors",

    y="Valuation (USD)"

)

plt.title("Investors vs Valuation")

plt.show()


# TOP 10 COUNTRIES

top_countries = (

    df["Country"]

    .value_counts()

    .head(10)

    .reset_index()

)

top_countries.columns = [

    "Country",

    "Count"

]

plt.figure(figsize=(12,6))

sns.barplot(

    data=top_countries,

    x="Count",

    y="Country"

)

plt.title("Top 10 Countries by Number of Startups")

plt.show()


# AVERAGE INVESTMENT BY INDUSTRY

industry_avg = (

    df.groupby("Industry")["Investment Amount (USD)"]

    .mean()

    .sort_values(ascending=False)

    .reset_index()

)

plt.figure(figsize=(12,6))

sns.barplot(

    data=industry_avg,

    x="Investment Amount (USD)",

    y="Industry"

)

plt.title("Average Investment by Industry")

plt.show()


# AVERAGE INVESTMENT BY YEAR FOUNDED

year_avg = ( df.groupby("Year Founded")["Investment Amount (USD)"].mean().reset_index() )

plt.figure(figsize=(12,6))

sns.lineplot(

    data=year_avg,

    x="Year Founded",

    y="Investment Amount (USD)",

    marker="o"

)

plt.title("Average Investment by Year Founded")

plt.show()


# CORRELATION HEATMAP

plt.figure(figsize=(8,6))

sns.heatmap(

    df[numeric_columns].corr(),

    annot=True,

    cmap="coolwarm",

    linewidths=0.5

)

plt.title("Correlation Heatmap")

plt.show()


# PAIR PLOT

sns.pairplot(

    df[numeric_columns]

)

plt.show()