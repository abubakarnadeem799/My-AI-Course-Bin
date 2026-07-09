# Real_Estate_Sales_2001-2022_GL

# IMPORT LIBRARY

import numpy as np


# LOAD DATASET

sale_amount, assessed_value, sales_ratio, list_year = np.genfromtxt( "Data_Sets/Data Science_Assignments/Real_Estate_Sales_2001-2022_GL-Short.csv", delimiter=",", skip_header=1, usecols=(6, 5, 7, 1), unpack=True,  dtype=float,  missing_values=["", "NA", "NaN", "NULL"],  filling_values=np.nan,  invalid_raise=False, encoding="utf-8" )


# SHAPE

print("\nShape")
print(sale_amount.shape)


# SIZE

print("\nSize")
print(sale_amount.size)


# DIMENSIONS

print("\nDimensions")
print(sale_amount.ndim)


# DATA TYPE

print("\nData Type")
print(sale_amount.dtype)


# TOTAL RECORDS

print("\nTotal Records")
print(len(sale_amount))


# MEAN SALE AMOUNT

print("\nMean Sale Amount")
print(np.nanmean(sale_amount))


# MEDIAN SALE AMOUNT

print("\nMedian Sale Amount")
print(np.nanmedian(sale_amount))


# MAXIMUM SALE AMOUNT

print("\nMaximum Sale Amount")
print(np.nanmax(sale_amount))


# MINIMUM SALE AMOUNT

print("\nMinimum Sale Amount")
print(np.nanmin(sale_amount))


# TOTAL SALE AMOUNT

print("\nTotal Sale Amount")
print(np.nansum(sale_amount))


# STANDARD DEVIATION

print("\nStandard Deviation")
print(np.nanstd(sale_amount))


# VARIANCE

print("\nVariance")
print(np.nanvar(sale_amount))


# MEAN ASSESSED VALUE

print("\nAverage Assessed Value")
print(np.nanmean(assessed_value))


# MEDIAN ASSESSED VALUE

print("\nMedian Assessed Value")
print(np.nanmedian(assessed_value))


# MAXIMUM ASSESSED VALUE

print("\nMaximum Assessed Value")
print(np.nanmax(assessed_value))


# MINIMUM ASSESSED VALUE

print("\nMinimum Assessed Value")
print(np.nanmin(assessed_value))


# TOTAL ASSESSED VALUE

print("\nTotal Assessed Value")
print(np.nansum(assessed_value))


# MEAN SALES RATIO

print("\nAverage Sales Ratio")
print(np.nanmean(sales_ratio))


# MAXIMUM SALES RATIO

print("\nMaximum Sales Ratio")
print(np.nanmax(sales_ratio))


# MINIMUM SALES RATIO

print("\nMinimum Sales Ratio")
print(np.nanmin(sales_ratio))


# UNIQUE LIST YEARS

print("\nUnique List Years")
print(np.unique(list_year))


# NUMBER OF UNIQUE YEARS

print("\nNumber of Unique Years")
print(np.unique(list_year).size)


# MEAN LIST YEAR

print("\nAverage List Year")
print(np.nanmean(list_year))


# PERCENTILES

percentiles = np.nanpercentile( sale_amount, [25, 50, 75] )

print("\n25th Percentile")
print(percentiles[0])

print("\n50th Percentile")
print(percentiles[1])

print("\n75th Percentile")
print(percentiles[2])


# RANGE OF SALE AMOUNT

print("\nRange of Sale Amount")
print( np.nanmax(sale_amount) - np.nanmin(sale_amount) )


# NUMBER OF MISSING VALUES

print("\nMissing Sale Amount")
print(np.isnan(sale_amount).sum())

print("\nMissing Assessed Value")
print(np.isnan(assessed_value).sum())

print("\nMissing Sales Ratio")
print(np.isnan(sales_ratio).sum())


# FIRST 10 SALE AMOUNTS

print("\nFirst 10 Sale Amounts")
print(sale_amount[:10])


# LAST 10 SALE AMOUNTS

print("\nLast 10 Sale Amounts")
print(sale_amount[-10:])