import pandas as pd

df = pd.read_csv('RealEstate-USA.csv', delimiter=",", parse_dates=[1], date_format={'date_added': '%d-%m-%Y'})


print(df)

print("df - data types" , df.dtypes)

print("df.info():   " , df.info() )

print('Last three rows:')
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


