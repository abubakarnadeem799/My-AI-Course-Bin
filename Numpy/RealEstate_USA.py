# RealEstate-USA.csv

# Numpy

# IMPORT LIBRARY

import numpy as np


# LOAD DATASET

price, bed, bath, acre_lot, house_size = np.genfromtxt( "Data_Sets/Data Science_Assignments/RealEstate-USA.csv", delimiter=",", skip_header=1, usecols=(2,3,4,5,10),  unpack=True, dtype=float, filling_values=np.nan, encoding="utf-8" )


# DISPLAY DATA

print("Price:\n", price)

print("Bedrooms:\n", bed)

print("Bathrooms:\n", bath)

print("Acre Lot:\n", acre_lot)

print("House Size:\n", house_size)


# SHAPE

print("\nShape of Price Array")

print(price.shape)


# SIZE

print("\nTotal Number of Records")

print(price.size)


# DIMENSIONS

print("\nNumber of Dimensions")

print(price.ndim)


# DATA TYPE

print("\nData Type")

print(price.dtype)


# MEAN PRICE

print("\nAverage House Price")

print(np.nanmean(price))


# MEDIAN PRICE

print("\nMedian House Price")

print(np.nanmedian(price))


# MAXIMUM PRICE

print("\nMaximum House Price")

print(np.nanmax(price))


# MINIMUM PRICE

print("\nMinimum House Price")

print(np.nanmin(price))


# STANDARD DEVIATION

print("\nPrice Standard Deviation")

print(np.nanstd(price))


# AVERAGE HOUSE SIZE

print("\nAverage House Size")

print(np.nanmean(house_size))


# MAXIMUM HOUSE SIZE

print("\nMaximum House Size")

print(np.nanmax(house_size))


# MINIMUM HOUSE SIZE

print("\nMinimum House Size")

print(np.nanmin(house_size))


# AVERAGE BEDROOMS

print("\nAverage Bedrooms")

print(np.nanmean(bed))


# AVERAGE BATHROOMS

print("\nAverage Bathrooms")

print(np.nanmean(bath))


# AVERAGE ACRE LOT

print("\nAverage Acre Lot")

print(np.nanmean(acre_lot))


# TOTAL PRICE

print("\nTotal Property Value")

print(np.nansum(price))


# TOTAL HOUSE SIZE

print("\nTotal House Size")

print(np.nansum(house_size))