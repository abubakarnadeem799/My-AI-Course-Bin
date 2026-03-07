print("Welcome to my Set practice")
Set_Customers = {"Ali", 500, 500.50, True}
print(Set_Customers)

Set_Customers.add("Jim")
print(Set_Customers)

Set_Customers.discard(500)
print(Set_Customers)

for item in Set_Customers:
    print(item)