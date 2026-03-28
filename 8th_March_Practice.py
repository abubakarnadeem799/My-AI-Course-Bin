Customer_Info =  "Customer name is Kim jo and he located in california"
print(len(Customer_Info))
count = 0
for i in  Customer_Info:
    count += 1
    print(i, "is at index no.", count)

print(Customer_Info.upper())
print(Customer_Info.lower())

name_str = Customer_Info[17:23:1]
print(name_str)

city_str = Customer_Info[-10:]
print(city_str)

city_str = Customer_Info[42:52:1]
print(city_str)



