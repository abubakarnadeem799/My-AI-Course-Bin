# FastFoodRestaurants.csv


# IMPORT LIBRARIES

import pandas as pd
import numpy as np

# PANDAS ANALYSIS

# Load Dataset

df = pd.read_csv("Data_Sets/Data Science_Assignments/FastFoodRestaurants.csv")


# Display First 5 Records

print(df.head())


# Display Last 5 Records

print(df.tail())


# Dataset Shape

print("Shape :", df.shape)


# Dataset Information

print(df.info())


# Column Names

print(df.columns)


# Data Types

print(df.dtypes)


# Missing Values

print(df.isnull().sum())


# Duplicate Records

print("Duplicate Rows :", df.duplicated().sum())


# Remove Duplicate Records

df = df.drop_duplicates()


# Statistical Summary

print(df.describe(include="all"))


# Unique Restaurant Names

print(df["name"].unique())


# Number of Restaurant Brands

print("Unique Restaurants :", df["name"].nunique())


# Unique Cities

print("Unique Cities :", df["city"].nunique())


# Unique States

print("Unique Provinces :", df["province"].nunique())


# Restaurant Count by Brand

restaurant_count = df.groupby("name").size().sort_values(ascending=False)

print(restaurant_count)


# Restaurant Count by State

province_count = df.groupby("province").size().sort_values(ascending=False)

print(province_count)


# Restaurant Count by City

city_count = df.groupby("city").size().sort_values(ascending=False)

print(city_count.head(10))


# Average Latitude and Longitude by Restaurant

location_analysis = df.groupby("name")[["latitude","longitude"]].mean()

print(location_analysis)


# Sort Restaurants Alphabetically

sorted_restaurants = df.sort_values("name")

print(sorted_restaurants.head())


# Sort by City

sorted_city = df.sort_values("city")

print(sorted_city.head())


# Filter McDonald's Restaurants

mcdonalds = df[df["name"]=="McDonald's"]

print(mcdonalds.head())


# Restaurants in New York

new_york = df[df["province"]=="NY"]

print(new_york.head())


# Restaurants in Ohio

ohio = df[df["province"]=="OH"]

print(ohio.head())


# Reset Index

df = df.reset_index(drop=True)

print(df.head())