# startup-growth-investment-data.csv

# NUMPY

# IMPORT LIBRARY

import numpy as np


# LOAD DATASET

funding_rounds, investment_amount, valuation, investors, year_founded, growth_rate = np.genfromtxt( "Data_Sets/Data Science_Assignments/startup_growth_investment_data.csv", delimiter=",",  skip_header=1,  usecols=(2,3,4,5,7,8),  unpack=True, dtype=float, filling_values=np.nan,  missing_values=["", "NA", "NaN"],  invalid_raise=False, encoding="utf-8" )


# SHAPE

print("\nShape")
print(investment_amount.shape)


# SIZE

print("\nSize")
print(investment_amount.size)


# DIMENSIONS

print("\nDimensions")
print(investment_amount.ndim)


# DATA TYPE

print("\nData Type")
print(investment_amount.dtype)


# TOTAL RECORDS

print("\nTotal Records")
print(len(investment_amount))


# MEAN INVESTMENT

print("\nMean Investment")
print(np.nanmean(investment_amount))


# MEDIAN INVESTMENT

print("\nMedian Investment")
print(np.nanmedian(investment_amount))


# MAXIMUM INVESTMENT

print("\nMaximum Investment")
print(np.nanmax(investment_amount))


# MINIMUM INVESTMENT

print("\nMinimum Investment")
print(np.nanmin(investment_amount))


# TOTAL INVESTMENT

print("\nTotal Investment")
print(np.nansum(investment_amount))


# STANDARD DEVIATION OF INVESTMENT

print("\nStandard Deviation of Investment")
print(np.nanstd(investment_amount))


# VARIANCE OF INVESTMENT

print("\nVariance of Investment")
print(np.nanvar(investment_amount))


# MEAN VALUATION

print("\nMean Valuation")
print(np.nanmean(valuation))


# MEDIAN VALUATION

print("\nMedian Valuation")
print(np.nanmedian(valuation))


# MAXIMUM VALUATION

print("\nMaximum Valuation")
print(np.nanmax(valuation))


# MINIMUM VALUATION

print("\nMinimum Valuation")
print(np.nanmin(valuation))


# TOTAL VALUATION

print("\nTotal Valuation")
print(np.nansum(valuation))


# AVERAGE FUNDING ROUNDS

print("\nAverage Funding Rounds")
print(np.nanmean(funding_rounds))


# MAXIMUM FUNDING ROUNDS

print("\nMaximum Funding Rounds")
print(np.nanmax(funding_rounds))


# MINIMUM FUNDING ROUNDS

print("\nMinimum Funding Rounds")
print(np.nanmin(funding_rounds))


# AVERAGE NUMBER OF INVESTORS

print("\nAverage Number of Investors")
print(np.nanmean(investors))


# MAXIMUM NUMBER OF INVESTORS

print("\nMaximum Number of Investors")
print(np.nanmax(investors))


# MINIMUM NUMBER OF INVESTORS

print("\nMinimum Number of Investors")
print(np.nanmin(investors))


# AVERAGE GROWTH RATE

print("\nAverage Growth Rate")
print(np.nanmean(growth_rate))


# MAXIMUM GROWTH RATE

print("\nMaximum Growth Rate")
print(np.nanmax(growth_rate))


# MINIMUM GROWTH RATE

print("\nMinimum Growth Rate")
print(np.nanmin(growth_rate))


# UNIQUE FOUNDED YEARS

print("\nUnique Founded Years")
print(np.unique(year_founded))


# NUMBER OF UNIQUE YEARS

print("\nNumber of Unique Years")
print(np.unique(year_founded).size)


# PERCENTILES OF INVESTMENT

percentiles = np.nanpercentile( investment_amount, [25, 50, 75] )

print("\n25th Percentile")
print(percentiles[0])

print("\n50th Percentile")
print(percentiles[1])

print("\n75th Percentile")
print(percentiles[2])


# RANGE OF INVESTMENT

print("\nRange of Investment")
print( np.nanmax(investment_amount) -  np.nanmin(investment_amount) )


# MISSING VALUES

print("\nMissing Investment Values")
print(np.isnan(investment_amount).sum())

print("\nMissing Valuation Values")
print(np.isnan(valuation).sum())


# FIRST 10 INVESTMENTS

print("\nFirst 10 Investment Records")
print(investment_amount[:10])


# LAST 10 INVESTMENTS

print("\nLast 10 Investment Records")
print(investment_amount[-10:])