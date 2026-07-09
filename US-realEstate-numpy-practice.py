#reale_state-property-data-By-Kaggle.csv
import numpy as np


price, acre_lot, street = np.genfromtxt('RealEstate-USA.csv', delimiter=',', usecols=(2,5,6), unpack=True, dtype= None, encoding='utf-8', invalid_raise=False,  skip_header=1 )


print(price)
print(acre_lot)
print(street )


# real_estate price  - statistics operations
print("real_estate  Price mean: " , np.mean(price))
print("real_estate  Price average: " , np.average(price))
print("real_estate  Price std: " , np.std(price))
print("real_estate  Price mod: " , np.median(price))
print("real_estate  Price percentile - 25: " , np.percentile(price,25))
print("real_estate  Price percentile  - 75: " , np.percentile(price,75))
print("real_estate  Price percentile  - 3: " , np.percentile(price,3))
print("real_estate  Price min : " , np.min(price))
print("real_estate  Price max : " , np.max(price))

# real_estate  - maths operations
print("real_estate Price square: " , np.square(price))
print("real_estate Price sqrt: " , np.sqrt(price))
print("real_estate Price pow: " , np.power(price,price))
print("real_estate Price abs: " , np.abs(price))



# Perform basic arithmetic operations
addition = acre_lot + price
subtraction = acre_lot - price
multiplication = acre_lot * price
division = acre_lot / price

print(" real_estate acre_lot - price - Addition:", addition)
print(" real_estate acre_lot - price - Subtraction:", subtraction)
print(" real_estate acre_lot - price - Multiplication:", multiplication)
print(" real_estate acre_lot- price - Division:", division)

#Trigonometric Functions

pricePie = (price/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("real_estate Price - div - pie  - Sine values:", sine_values)
print("real_estate Price - div - pie Cosine values:", cosine_values)
print("real_estate Price - div - pie Tangent values:", tangent_values)

print("real_estate Price - div - pie  - Exponential values:", np.exp(pricePie))


# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("real_estate Price - div - pie  - Natural logarithm values:", log_array)
print("real_estate Price - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(pricePie)
print("real_estate Price - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)
print("real_estate Price - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(pricePie)
print("real_estate Price - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("real_estate Price - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("real_estate Price - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#real_estate acre_lot Plus price - 2 dimentional arrary
D2acre_lotprice = np.array([acre_lot, price])

print ("real_estate acre_lot Plus price - 2 dimentional arrary - " ,D2acre_lotprice)

# check the dimension of array1
print("real_estate acre_lot Plus price  - 2 dimentional arrary - dimension" , D2acre_lotprice.ndim) 
# Output: 2

# return total number of elements in array1
print("real_estate acre_lot Plus price - 2 dimentional arrary - total number of elements" ,D2acre_lotprice.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("real_estate acre_lot Plus price - 2 dimentional arrary - gives size of array in each dimension" ,D2acre_lotprice.shape)
# Output: (2,3)

# check the data type of array1
print("real_estate acre_lot Plus price - 2 dimentional arrary - data type" ,D2acre_lotprice.dtype) 
# Output: int64

# Splicing array
D2LongLatSlice=  D2LongLat[:1,:5]
print("Zameen.com Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2acre_lotpriceSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("Zameen.com Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2acre_lotpriceSlice2)



# Indexing array
D2LongLatSliceItemOnly=  D2LongLatSlice[0,1]
print("real_estate acre_lot Plus price  - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2acre_lotpriceSliceItemOnly)
D2LongLatSlice2ItemOnly=  D2LongLatSlice2[0, 2]
print("real_estate acre_lot Plus price  - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " ,D2acre_lotpriceSlice2ItemOnly)


#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(D2acre_lotpricet):
    print(elem)

#EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(D2acre_lotprice):
    print(index, elem)


# 2 x 149 ========>>>>> 1  x 298 - reshape
D2LongLat1TO298 = np.reshape(D2LongLat, (1, 298))
print("real_estate acre_lot Plus price  - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : " , D2acre_lotprice1TO298)
print("real_estate acre_lot Plus price - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : Size " , D2acre_lotprice1TO298.size)
print("real_estate acre_lot Plus price  - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : ndim " , D2acre_lotprice1TO298.ndim)
print("real_estate acre_lot Plus price  - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : shape " , D2acre_lotprice1TO298.shape)
print("real_estate acre_lot Plus price  - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : ndim " , D2acre_lotprice1TO298.ndim)




print()
