print("Welcome to my Python Practice")

Customer_List = ["Trump", 200, 0.0200, True ]
print(Customer_List)
print(type(Customer_List))

Customer_List.append("Ayotollah ALi Khaninae")
print(Customer_List)

Customer_List.insert(1, "Tahran")
print(Customer_List)

print(Customer_List[1])
print(Customer_List[2])
print(Customer_List)
print(type(Customer_List[2]))

Customer_List.remove(200)
print(Customer_List)

Customer_List.pop(1)
print(Customer_List)

Customer_List[0]= "My Change Value"
print(Customer_List)
