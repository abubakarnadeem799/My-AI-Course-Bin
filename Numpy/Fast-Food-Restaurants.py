# Fast-Food-Restaurants

import numpy as np

latitude, longitude, postalCode = np.genfromtxt('Data_Sets/Data Science_Assignments/FastFoodRestaurants.csv', delimiter=",", usecols=(4,5,7), unpack=True, dtype=str, skip_header=1,invalid_raise=False,
    encoding="utf-8" )


print(latitude)
print(longitude)
print(postalCode)

# Convert numeric columns to float
latitude = latitude.astype(float)
longitude = longitude.astype(float)

# Statistical Operations

print("\nLatitude Mean:", np.mean(latitude))
print("Latitude Average:", np.average(latitude))
print("Latitude Median:", np.median(latitude))
print("Latitude Standard Deviation:", np.std(latitude))

print("Latitude Minimum:", np.min(latitude))
print("Latitude Maximum:", np.max(latitude))

print("25th Percentile:", np.percentile(latitude,25))
print("50th Percentile:", np.percentile(latitude,50))
print("75th Percentile:", np.percentile(latitude,75))

print("Variance:", np.var(latitude))
print("Sum:", np.sum(latitude))


# Mathematical Operations

print("\nSquare")
print(np.square(latitude))

print("\nSquare Root")
print(np.sqrt(np.abs(latitude)))

print("\nAbsolute")
print(np.abs(latitude))

print("\nRound")
print(np.round(latitude,2))

print("\nFloor")
print(np.floor(latitude))

print("\nCeil")
print(np.ceil(latitude))


# Arithmetic Operations

addition = latitude + longitude
subtraction = latitude - longitude
multiplication = latitude * longitude
division = latitude / longitude

print("\nAddition")
print(addition)

print("\nSubtraction")
print(subtraction)

print("\nMultiplication")
print(multiplication)

print("\nDivision")
print(division)


# Trigonometric Functions

lat = (latitude / np.pi) + 1

print("\nSin")
print(np.sin(lat))

print("\nCos")
print(np.cos(lat))

print("\nTan")
print(np.tan(lat))


# Exponential & Logarithmic Functions

print("\nExponential")
print(np.exp(lat))

print("\nNatural Log")
print(np.log(lat))

print("\nLog Base 10")
print(np.log10(lat))


# Hyperbolic Functions

print("\nSinh")
print(np.sinh(lat))

print("\nCosh")
print(np.cosh(lat))

print("\nTanh")
print(np.tanh(lat))

print("\nInverse Sinh")
print(np.arcsinh(lat))

print("\nInverse Cosh")
print(np.arccosh(lat))


# Create 2D Array

restaurant_coordinates = np.array([
    latitude,
    longitude
])

print(restaurant_coordinates)


# Array Properties

print("Dimensions:", restaurant_coordinates.ndim)

print("Shape:", restaurant_coordinates.shape)

print("Size:", restaurant_coordinates.size)

print("Data Type:", restaurant_coordinates.dtype)


# Array Slicing

slice1 = restaurant_coordinates[:1,:5]

print(slice1)

slice2 = restaurant_coordinates[:2,5:15]

print(slice2)


# Indexing

print(slice1[0,2])

print(slice2[1,4])


# Iterate without index

for value in np.nditer(restaurant_coordinates):
    print(value)


# Iterate with index

for index, value in np.ndenumerate(restaurant_coordinates):
    print(index, value)

    
# Reshape Array

reshape_array = np.reshape(
    restaurant_coordinates,
    (1, restaurant_coordinates.size)
)

print(reshape_array)

print("Shape:", reshape_array.shape)

print("Size:", reshape_array.size)

print("Dimensions:", reshape_array.ndim)

# Sorting
np.sort(latitude)

# Unique values
np.unique(city)

# Count values
len(np.unique(city))

# Correlation between Latitude & Longitude
np.corrcoef(latitude, longitude)

# Dot Product
np.dot(latitude[:100], longitude[:100])

# Cumulative Sum
np.cumsum(latitude)

# Cumulative Product
np.cumprod(np.abs(latitude[:20]))

# Difference
np.diff(latitude)

# Clip values
np.clip(latitude, 25, 50)

# Boolean Filtering
latitude[latitude > 40]

# Any / All
np.any(latitude > 50)
np.all(latitude > 20)

# Argmax / Argmin
np.argmax(latitude)
np.argmin(latitude)

# Mean Centering
latitude - np.mean(latitude)

# Normalization
(latitude - np.min(latitude)) / (np.max(latitude) - np.min(latitude))



