#Health care: Heart attack possibility-data-By-Kaggle.csv
import numpy as np

age, sex, cp, trestbps,	chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target = np.genfromtxt('MachineLearning/heart.csv', delimiter=',', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack=True, dtype=None,skip_header=1)

print(age)
print(sex)
print(cp)
print(chol)
print(fbs)
print(restecg)
print(thalach)
print(exang)
print(oldpeak)
print(slope)
print(ca)
print(thal)
print(target)


# Heart attack possibility age  - statistics operations
print("Health Care age mean: " , np.mean(age))
print("Health Care age average: " , np.average(age))
print("Health Care age std: " , np.std(age))
print("Health Care age mod: " , np.median(age))
print("Health Care age percentile - 25: " , np.percentile(age,25))
print("Health Care age percentile  - 75: " , np.percentile(age,75))
print("Health Care age percentile  - 3: " , np.percentile(age,3))
print("Health Care age min : " , np.min(age))
print("Health Care age max : " , np.max(age))

# Health Care age  - maths operations
print("Health Care age square: " , np.square(age))
print("Health Care age sqrt: " , np.sqrt(age))
print("Health Care age pow: " , np.power(age, age))
print("Health Care age abs: " , np.abs(age))



# Perform basic arithmetic operations
addition = chol + thalach
subtraction = chol - thalach
multiplication = chol * thalach
division = chol / thalach

print(" Health Care chol - thalach - Addition:", addition)
print(" Health Care chol - thalach - Subtraction:", subtraction)
print(" Health Care chol - thalach- Multiplication:", multiplication)
print(" Health Care chol - thalach - Division:", division)


#Trigonometric Functions

agePie = (age/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(agePie)
cosine_values = np.cos(agePie)
tangent_values = np.tan(agePie)

print("Health Care age - div - pie  - Sine values:", sine_values)
print("Health Care age - div - pie Cosine values:", cosine_values)
print("Health Care age - div - pie Tangent values:", tangent_values)

print("Health Care age - div - pie  - Exponential values:", np.exp(agePie))


# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(agePie)
log10_array = np.log10(agePie)

print("Health Care age - div - pie  - Natural logarithm values:", log_array)
print("Health Care age - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(agePie)
print("Health Care age - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(agePie)
print("Health Care age - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(agePie)
print("Health Care age - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(agePie)
print("Health Care age - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(agePie)
print("Health Care age - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#Health Care  thalach Pluschol - 2 dimentional arrary
D2thalachchol = np.array([chol,
                  thalach])

print ("Health Care  thalach Plus chol - 2 dimentional arrary - " ,D2thalachchol)

# check the dimension of array1
print("Health Care  thalach Plus chol - 2 dimentional arrary - dimension" , D2thalachchol.ndim) 
# Output: 2

# return total number of elements in array1
print("Health Care  thalach Plus chol - 2 dimentional arrary - total number of elements" ,D2thalachchol.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("Health Care  thalach Plus chol - 2 dimentional arrary - gives size of array in each dimension" ,D2thalachchol.shape)
# Output: (2,3)

# check the data type of array1
print("Health Care  thalach Plus chol - 2 dimentional arrary - data type" ,D2thalachchol.dtype) 
# Output: int64

# Splicing array
D2thalachcholSlice=  D2thalachchol[:1,:5]
print("Health Care  thalach Plus chol - 2 dimentional arrary - Splicing array - D2 thalachchol[:1,:5] " , D2thalachcholSlice)
D2thalachcholSlice2=  D2thalachchol[:1, 4:15:4]
print("Health Care  thalach Plus chol - 2 dimentional arrary - Splicing array - D2 thalachchol[:1, 4:15:4] " , D2thalachcholSlice2)



# Indexing array
D2thalachcholSliceItemOnly=  D2thalachcholSlice[0,1]
print("Health Care  thalach Pluschol - 2 dimentional arrary - Index array - D2 thalachcholSlice[1,5] " , D2thalachcholSliceItemOnly)
D2thalachcholSlice2ItemOnly=  D2thalachcholSlice2[0, 2]
print("Health Care  thalach Pluschol - 2 dimentional arrary - index array - D2 thalachcholSlice2[0, 2] " , D2thalachcholSlice2ItemOnly)


#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(D2thalachchol):
    print(elem)

#EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(D2thalachchol):
    print(index, elem)

"""# for loop
rows = np.shape(D2 thalachchol[0])[0]
cols = np.shape(D2 thalachchol[1])[0]
for i in range(0, (rows + 1)):
    for j in range(0, (cols + 1)):
        print (D2 thalachchol[i,j])
"""


# 2 x 303 ========>>>>> 1  x 606 - reshape
D2thalachchol1TO298 = np.reshape(D2thalachchol, (1, 606))
print("Health Care  thalach Pluschol - 2 dimentional arrary - np.reshape(D2 thalachchol, (1, 606)) : " , D2thalachchol1TO298)
print("Health Care  thalach Pluschol - 2 dimentional arrary - np.reshape(D2 thalachchol, (1, 606)) : Size " , D2thalachchol1TO298.size)
print("Health Care  thalach Pluschol - 2 dimentional arrary - np.reshape(D2 thalachchol, (1, 606)) : ndim " , D2thalachchol1TO298.ndim)
print("Health Care  thalach Pluschol - 2 dimentional arrary - np.reshape(D2 thalachchol, (1, 606)) : shape " , D2thalachchol1TO298.shape)
print("Health Care  thalach Pluschol - 2 dimentional arrary - np.reshape(D2 thalachchol, (1, 298)) : ndim " , D2thalachchol1TO298.ndim)

print()

