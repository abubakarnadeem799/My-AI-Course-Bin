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

#Health Care-Heart Attrack-data-By-Kaggle.csv
import pandas as pd

#  Read csv file to DataFrame
#  Reference: https://pandas.pydata.org/docs/dev/reference/api/pandas.read_csv.html
#  Note below, date formatting - In Pandas, DateTime is a data type that represents a single point in time. It is especially useful when dealing with time-series data like stock prices, weather records, economic indicators etc.
df = pd.read_csv('MachineLearning/heart.csv',delimiter=",",parse_dates=[13], date_format={'date_added': '%d-%m-%Y'})

print(df)

print("df - data types" , df.dtypes)

print("df.info():   " , df.info() )

# display the last three rows
print('Last three Rows:')
print(df.tail(3))

# display the first three rows
print('First Three Rows:')
print(df.head(3))
print()

#Summary of Statistics of DataFrame using describe() method.
print("Summary of Statistics of DataFrame using describe() method", df.describe())

#Counting the rows and columns in DataFrame using shape(). It returns the no. of rows and columns enclosed in a tuple.
print("Counting the rows and columns in DataFrame using shape() : " ,df.shape)
print()



# access the Name column
trestbps = df['trestbps']
print("access the Name column: df : ")
print(trestbps)
print()

# access multiple columns
trestbps_chol = df[['trestbps','chol']]
print("access multiple columns: df : ")
print(trestbps_chol)
print()



# Case 1 : using .loc - default case - starts here
# Reference: https://www.datacamp.com/tutorial/loc-vs-iloc
# 
"""
Syntax               df.loc[row_indexer, column_indexer]              df.iloc[row_indexer, column_indexer]
Indexing Method      Label-based                                      Position-based indexing
Used for Reference   Row and column labels (names)                    Numerical indices of rows and columns (starting from 0)
"""
#Selecting a single row using .loc
second_row = df.loc[1]
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
second_row2 = df.loc[[1, 3]]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

#Selecting a slice of rows using .loc
second_row3 = df.loc[1:5]
print("#Selecting a slice of rows using .loc")
print(second_row3)
print()


#Conditional selection of rows using .loc
second_row4 = df.loc[df['trestbps'] == 'Gateway Properties']
print("#Conditional selection of rows using .loc")
print(second_row4)
print()

#Selecting a single column using .loc
second_row5 = df.loc[:1,'trestbps']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df.loc[:1,['trestbps','chol']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting a slice of columns using .loc
second_row7 = df.loc[:1,'thalach':'trestbps']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df.loc[df['trestbps'] == 'Gateway Properties','thalach':'trestbps']
print("#Combined row and column selection using .loc")
print(second_row8)
print()
# Case 1 : using .loc - default case - ends here


print("# Case 2 : using .loc with index_col - starts here")
# Case 2 : using .loc with index_col - starts here
# Second cycle - with index_col as cp
# Why Second cycle - Note Index - , index_col='cp'
df_index_col = pd.read_csv('MachineLearning/heart.csv',delimiter=",",parse_dates=[13], date_format={'date_added': '%d-%m-%Y'} , index_col='cp' )

print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())
# Second cycle - with index_col as property_id

#Selecting a single row using .loc
second_row = df_index_col.loc[3]
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
second_row2 = df_index_col.loc[[3, 1]]
print("#Selecting multiple rows using .loc")
print(second_row2)
print()

#Conditional selection of rows using .loc
second_row3 = df_index_col.loc[df_index_col['trestbps'] == 'Gateway Properties']
print("#Conditional selection of rows using .loc")
print(second_row3)
print()




#Combined row and column selection using .loc
second_row8 = df_index_col.loc[df_index_col['trestbps'] == 'Gateway Properties','thalach':'trestbps']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

# Case 2 : using .loc with index_col  -  ends here


print("# Case 3 : Using .iloc - starts here")
# Case 3 : Using .iloc - starts here
"""Using .iloc: Selection by Integer Position
.iloc selects by position instead of label. This is the standard syntax of using .iloc: df.iloc[row_indexer, column_indexer]. There are two special things to look out for:

Counting starting at 0: The first row and column have the index 0, the second one index 1, etc.
Exclusivity of range end value: When using a slice, the row or column specified behind the colon is not included in the selection."""

#Selecting a single row using .iloc
second_row = df_index_col.iloc[0]
print("#Selecting a single row using .iloc")
print(second_row)
print()

#Selecting multiple rows using .iloc
second_row2 = df_index_col.iloc[[1, 3,5]]
print("#Selecting multiple rows using .iloc")
print(second_row2)
print()

#Selecting a slice of rows using .iloc
second_row3 = df_index_col.iloc[2:5]
print("#Selecting a slice of rows using .iloc")
print(second_row3)
print()

#Selecting a single column using .iloc
second_row5 = df_index_col.iloc[:,2]
print("#Selecting a single column using .iloc")
print(second_row5)
print()

#Selecting multiple columns using .iloc
second_row6 = df_index_col.iloc[:,[2,4]]
print("#Selecting multiple columns using .iloc")
print(second_row6)
print()

#Selecting a slice of columns using .iloc
second_row7 = df_index_col.iloc[:,2:4]
print("#Selecting a slice of columns using .iloc")
print(second_row7)
print()

#Combined row and column selection using .iloc
second_row8 = df_index_col.iloc[[1, 3,5],2:4]
print("#Combined row and column selection using .iloc")
print(second_row8)
print()

# Case 3 : Using .iloc - ends here

# Next Run 
print("Next Run")

""""Pandas DataFrame Manipulation
DataFrame manipulation in Pandas involves editing and modifying existing DataFrames. Some common DataFrame manipulation operations are:

Adding rows/columns
Removing rows/columns
Renaming rows/columns"""

#Add a New Row to a Pandas DataFrame
# add a new row
# Copy array from list and add to DataFrame
# 3477952;82;"https://www.zameen.com/Property/lahore_model_town_6_kanal_excellent_house_for_sale_in_model_town-347795-8-12.html";"House2";2200000002;"Model Town2";"Lahore2";"Punjab2";312.483868658082;742.325685501099;02;"6 Kanal2";"For Sale2";02;"07-17-2019";"Real Biz International2";"Usama Khan2"

df.loc[len(df.index)] = [63,1,3,145,233,1,0,150,0,2.3,0,0,1,1]
print("Modified DataFrame - add a new row:")
print(df)
print()


#Remove Rows/Columns from a Pandas DataFrame


# delete row with index 1
df.drop(1, axis=0, inplace=True)
# delete row with index 1
df.drop(index=2, inplace=True)
# delete rows with index 3 and 5
df.drop([3, 5], axis=0, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame - Remove Rows:")
print(df)

# delete age column
df.drop('cp', axis=1, inplace=True)
# delete marital status column
df.drop(columns='age', inplace=True)
# delete height and profession columns
df.drop(['thalach', 'exang'], axis=1, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame -  age , thalach , exang:")
print(df)


#Rename Labels in a DataFrame
# rename column 'Name' to 'First_Name'
df.rename(columns= {'province_name': 'province_nameChanged'}, inplace=True)
# rename columns 'Age' and 'City'
df.rename(mapper= {'bedrooms': 'bedrooms_Changed', 'date_added':'date_added_Changed'}, axis=1, inplace=True)
# display the DataFrame after renaming column
print("Modified DataFrame  - Rename Labels :")
print(df)


#Example: Rename Row Labels
# rename column one index label
df.rename(index={0: 7}, inplace=True)
# rename columns multiple index labels
df.rename(mapper={1: 10, 2: 100}, axis=0, inplace=True)
# display the DataFrame after renaming column
print("Modified DataFrame - Rename Row - 0  >>> 7 , 1 >>> 10 , 2 >>> 100  Labels:")
print(df)



#query() to Select Data
#The query() method in Pandas allows you to select data using a more SQL-like syntax.

# selected_rows = df.query('age > 25')
# Remove extra spaces from column names
df.columns = df.columns.str.strip()



"""
Reference: https://www.geeksforgeeks.org/python-pandas-dataframe-to_string/
Explanation: This code creates a DataFrame from a dictionary with three columns (Weight, Name, Age), structures it into a tabular format using pd.DataFrame() and converts it into a fully visible string representation with df.to_string().

Syntax
DataFrame.to_string(buf=None, columns=None, col_space=None, header=True, index=True, na_rep=’NaN’, formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal=’.’, line_width=None)


Parameters:

buf: Buffer to write the output string to (e.g., a file). Defaults to None, which means the output is returned as a string.
columns: Specifies a subset of columns to include in the output. If None, all columns are printed.
col_space: Defines the minimum width of each column.
header: Whether to print column names. Can also accept a list of column name aliases.
index: Whether to include index labels. Default is True.
na_rep: String representation for missing values (NaN). Default is ‘NaN’.
formatters: Dictionary or list of functions to apply to columns for formatting their output.
float_format: Formatter function to apply specifically to floating-point numbers.
sparsify: Controls hierarchical index formatting. If False, prints every multi-index key at each row.
index_names: Whether to print index names. Default is True.
justify: Alignment of column headers (‘left’, ‘right’, ‘center’, ‘justify’ or ‘justify-all’).
max_rows: Maximum number of rows to display. If exceeded, truncates output.
max_cols: Maximum number of columns to display. If exceeded, truncates output.
show_dimensions: If True, displays the shape (rows x columns) of the DataFrame.
decimal: Specifies the character for decimal separation (e.g., ‘,’ for European formatting).
line_width: Defines the maximum character width of a row before wrapping text."""



#Pandas groupby
#In Pandas, the groupby operation lets us group data based on specific columns. This means we can divide a DataFrame into smaller groups based on the values in these columns.

# group the DataFrame by the location_id column and
# calculate the sum of price for each category
grouped = df.groupby('oldpeak')['slope'].sum()

print(grouped.to_string())
print("grouped :" , len(grouped))


""""Pandas Data Cleaning
Data cleaning means fixing and organizing messy data. Pandas offers a wide range of tools and functions to help us clean and preprocess our data effectively.
"""
# use dropna() to remove rows with any missing values
df_cleaned = df.dropna()
print("Cleaned Data:\n",df_cleaned)


# filling NaN values with 0
df.fillna(0, inplace=True)

print("\nData after filling NaN with 0:\n", df)



# create a list named data
data = [2, 4, 6, 8]
# create Pandas array using data
array1 = pd.array(data)
print(array1)
"""<IntegerArray>
[2, 4, 6, 8]
Length: 4, dtype: Int64"""


# creating a pandas.array of integers
int_array = pd.array([1, 2, 3, 4, 5], dtype='int')
print(int_array)
print()

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#https://seaborn.pydata.org/generated/seaborn.set_theme.html
#https://seaborn.pydata.org/tutorial/aesthetics.html
#https://python-charts.com/seaborn/themes/
"""
 Built-in Themes
Seaborn provides five built-in themes:
darkgrid: Adds a gray background with white gridlines. It is the default theme.
whitegrid: Adds gray gridlines on a white background.
dark: Similar to darkgrid but without the gridlines.
white: Similar to whitegrid but without the gridlines.
ticks: Adds ticks to the axes and uses a white background.
Setting Themes
The seaborn.set_theme() or seaborn.set_style() function can be used to set the theme for all plots. """

# Sample data
data = pd.DataFrame({'x': np.arange(100), 'y': np.random.rand(100).cumsum()})

# Set the theme
sns.set_theme(style='darkgrid')
# Alternatively
# sns.set_style('darkgrid')

# Create a plot
sns.lineplot(x='x', y='y', data=data)
plt.show()

# Other themes can be set similarly
sns.set_theme(style='whitegrid')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='dark')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='white')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='ticks')
sns.lineplot(x='x', y='y', data=data)
plt.show()


"""Customizing Themes
It is possible to customize the themes further by passing a dictionary of parameters to the rc argument of seaborn.set_theme() or seaborn.set_style(). This allows for fine-grained control over the appearance of plots."""

# Customize the theme
sns.set_theme(style='darkgrid', rc={'axes.facecolor': 'grey', 'grid.color': 'white'})

# Create a plot
sns.lineplot(x='x', y='y', data=data)
plt.show()

"""seaborn.set_theme() allows customization of the appearance of plots by modifying matplotlib's rc parameters. It accepts a dictionary rc to override default settings. Here's a breakdown of commonly used rc parameters:
axes.facecolor: Background color of the plotting area (e.g., 'white', '#EAEAF2').
axes.edgecolor: Color of the axes lines (e.g., 'black', 'gray').
axes.linewidth: Width of the axes lines in points.
axes.grid: Whether to show the grid ('True' or 'False').
axes.grid.axis: Which axes to show the grid lines on ('x', 'y', or 'both').
axes.grid.which: Which grid lines to draw ('major', 'minor', or 'both').
axes.labelcolor: Color of the axis labels.
axes.labelsize: Size of the axis labels in points or as a relative string (e.g., 'large', 'small').
axes.titlesize: Size of the plot title.
xtick.color: Color of the x-axis tick marks and labels.
ytick.color: Color of the y-axis tick marks and labels.
xtick.labelsize: Size of the x-axis tick labels.
ytick.labelsize: Size of the y-axis tick labels.
grid.color: Color of the grid lines.
grid.linewidth: Width of the grid lines.
font.family: Font family to use (e.g., 'sans-serif', 'serif', 'monospace').
font.size: Default font size for text elements.
lines.linewidth: Width of lines in plots.
lines.linestyle: Style of lines (e.g., '-', '--', '-.', ':').
patch.edgecolor: Color of patch edges (e.g., in histograms, bar plots).
patch.linewidth: Width of patch edges.
legend.frameon: Whether to display a frame around the legend ('True' or 'False').
legend.fontsize: Size of the legend text.
figure.figsize: Size of the figure (width, height) in inches.
figure.facecolor: Background color of the entire figure."""

#Zameencom data - based examples
# Load data from a CSV file
df = pd.read_csv('Week4/zameencom-property-data-By-Kaggle-short.csv',delimiter=";", parse_dates=[14],  date_format={'date_added': '%m-%d-%Y'} , index_col='property_id')

print(df.dtypes)
dffilter= df.head(40)
dffilter100= df.head(100)
# https://seaborn.pydata.org/api.html#distribution-api

"""Distribution plots
displot    -  Figure-level interface for drawing distribution plots onto a FacetGrid.

histplot   -  Plot univariate or bivariate histograms to show distributions of datasets.

kdeplot    -  Plot univariate or bivariate distributions using kernel density estimation.

ecdfplot   -  Plot empirical cumulative distribution functions.

rugplot    -  Plot marginal distributions by drawing ticks along the x and y axes."""


#https://seaborn.pydata.org/generated/seaborn.set_theme.html
#https://seaborn.pydata.org/tutorial/aesthetics.html
#https://seaborn.pydata.org/tutorial/color_palettes.html

sns.set(style="whitegrid")


#https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot
"""This function provides access to several approaches for visualizing the univariate or bivariate distribution of data, including subsets of data defined by semantic mapping and faceting across multiple subplots. The kind parameter selects the approach to use:

histplot() (with kind="hist"; the default)

kdeplot() (with kind="kde")

ecdfplot() (with kind="ecdf"; univariate-only)"""

#kind='hist'  
g=sns.displot(data=dffilter, x="agency" , y="price" , hue="agent",  kind='hist'  )
g.figure.suptitle("sns.displot(data=dffilter, x=agency , y=price , hue=agent,  kind='hist'  )"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()


""""kind="kde" in Seaborn specifies the use of Kernel Density Estimation plots. KDE plots visualize the probability density of a continuous variable. Instead of discrete bins like in histograms, KDE plots use a continuous curve to estimate the underlying distribution of the data. This provides a smoother and often more informative representation of the data's distribution, especially for continuous variables."""
#kind='kde'
g=sns.displot(data=dffilter, x="price" , y="date_added" , kind='kde'  )
g.figure.suptitle("sns.displot(data=dffilter, x=price , y=date_added , kind='kde'  )"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()

#https://seaborn.pydata.org/generated/seaborn.kdeplot.html
#kind='kde'
g=sns.kdeplot(data=dffilter, x="price")
g.figure.suptitle("sns.kdeplot(data=dffilter, x=price)"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()


# See: https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn.histplot
"""Plot univariate or bivariate histograms to show distributions of datasets.
A histogram is a classic visualization tool that represents the distribution of one or more variables by counting the number of observations that fall within discrete bins."""
g = sns.histplot(data=dffilter, x='agency', y='price', hue='agency', multiple="stack")
g.figure.suptitle("sns.histplot(data=dffilter, x='agency', y='price', hue='agency', multiple=stack)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()

#https://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot
"""Draw a scatter plot with possibility of several semantic groupings.

The relationship between x and y can be shown for different subsets of the data using the hue, size, and style parameters. These parameters control what visual semantics are used to identify the different subsets. It is possible to show up to three dimensions independently by using all three semantic types, but this style of plot can be hard to interpret and is often ineffective. Using redundant semantics (i.e. both hue and style for the same variable) can be helpful for making graphics more accessible."""
# Use Seaborn to create a plot
g = sns.scatterplot(x='agency', y='price', data=dffilter)
g.figure.suptitle("sns.scatterplot(x='agency', y='price', data=dffilter)"  )
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()


#https://seaborn.pydata.org/generated/seaborn.lineplot.html
"""Draw a line plot with possibility of several semantic groupings.

The relationship between x and y can be shown for different subsets of the data using the hue, size, and style parameters. These parameters control what visual semantics are used to identify the different subsets. It is possible to show up to three dimensions independently by using all three semantic types, but this style of plot can be hard to interpret and is often ineffective. Using redundant semantics (i.e. both hue and style for the same variable) can be helpful for making graphics more accessible."""
g=sns.lineplot(data=dffilter, x="agency" , y="price"  )
g.figure.suptitle("sns.lineplot(data=dffilter, x=agency , y=price  )"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()



#https://seaborn.pydata.org/generated/seaborn.barplot.html
"""Show point estimates and errors as rectangular bars.

A bar plot represents an aggregate or statistical estimate for a numeric variable with the height of each rectangle and indicates the uncertainty around that estimate using an error bar. Bar plots include 0 in the axis range, and they are a good choice when 0 is a meaningful value for the variable to take."""
g=sns.barplot(data=dffilter, x="agency", y="price", legend=False)
g.figure.suptitle("sns.barplot(data=dffilter, x=agency, y=price, legend=False)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()


#https://seaborn.pydata.org/generated/seaborn.catplot.html
""""Figure-level interface for drawing categorical plots onto a FacetGrid.

This function provides access to several axes-level functions that show the relationship between a numerical and one or more categorical variables using one of several visual representations. The kind parameter selects the underlying axes-level function to use."""

g=sns.catplot(data=dffilter, x="agency", y="price")
g.figure.suptitle("sns.catplot(data=df, x=agency, y=price)"  )
# Display the plot
g.figure.show() 
read = input("Wait for me....")
#g.figure.clear()




#https://seaborn.pydata.org/generated/seaborn.heatmap.html
""""Plot rectangular data as a color-encoded matrix.

This is an Axes-level function and will draw the heatmap into the currently-active Axes if none is provided to the ax argument. Part of this Axes space will be taken and used to plot a colormap, unless cbar is False or a separate Axes is provided to cbar_ax."""
#.pivot(index="Model", columns="agency", values="price")
glue = dffilter.pivot(columns="agency", values="price")

g=sns.heatmap(glue)
g.figure.suptitle("sns.heatmap(glue)  - glue = dffilter.pivot(columns=agency, values=price)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()



""""Seaborn provides a variety of plot types for different data visualization needs. These can be broadly categorized as: 

Relational plots: 

    These plots visualize the relationship between two or more variables. 

    scatterplot(): Displays the relationship between two numerical variables using points. 

    lineplot(): Shows the relationship between two numerical variables as a line, often used for time series data or trends. 

Categorical plots: 

    These plots display the distribution of a variable across different categories. 

    barplot(): Shows the mean of a numerical variable for different categories, with error bars. 

    countplot(): Displays the frequency of each category. 

    boxplot(): Shows the distribution of a numerical variable for different categories using quartiles. 

    violinplot(): Similar to a boxplot, but also shows the probability density of the data. 

    stripplot(): Displays individual data points for each category, allowing for visualization of the distribution. 

    swarmplot(): Similar to a stripplot, but the points are adjusted to avoid overlapping, providing a better view of the distribution. 

Distribution plots: 

    These plots visualize the distribution of a single variable. 

    histplot(): Displays the frequency distribution of a numerical variable using bins. 

    kdeplot(): Shows the estimated probability density function of a numerical variable. 

    ecdfplot(): Displays the empirical cumulative distribution function of a numerical variable. 

Multi-plot grids: 

    These functions allow for creating multiple plots at once, useful for comparing relationships between multiple variables. 

    pairplot(): Creates a matrix of scatter plots for all pairs of variables in a dataset. 

    relplot(): Facets scatterplot() and lineplot() across additional categorical variables. 

    catplot(): Facets categorical plots across additional categorical variables. 

Regression plots: 

    These plots visualize the relationship between two variables and fit a regression model. 

    lmplot(): Combines regplot() with FacetGrid to show linear relationships with the option to condition on other variables. 

    regplot(): Shows the relationship between two variables with a fitted regression line. 

Other plots: 

    heatmap(): Displays a matrix of values as a heatmap, often used for correlation matrices or other tabular data. 


Note: The choice of plot type depends on the type of data and the insights you want to extract. For comparing distributions, boxplot() or violinplot() are suitable. For visualizing relationships between two variables, scatterplot() or lineplot() can be used. For exploring relationships between multiple variables, pairplot() is a powerful tool. 

 """

