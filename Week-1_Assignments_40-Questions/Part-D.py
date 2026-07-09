# Part D - Python Dictionaries

# Question 1
student = {
    "name": "Ali",
    "age": 25
}

print("Question 1")
print(student["name"])


# Question 2
student["city"] = "Lahore"

print("\nQuestion 2")
print(student)


# Question 3
student = {
    "name": "Ali",
    "age": 25
}

student["age"] = 30

print("\nQuestion 3")
print(student)


# Question 4
student = {
    "name": "Ali",
    "age": 25
}

del student["age"]

print("\nQuestion 4")
print(student)


# Question 5
employee = {
    "name": "Ali",
    "age": 30
}

print("\nQuestion 5")
print("Is 'salary' present?", "salary" in employee)


# Question 6
data = {
    "a": 1,
    "b": 2
}

print("\nQuestion 6")
print(data.keys())


# Question 7
print("\nQuestion 7")
print(data.values())


# Question 8
data = {
    "x": 10,
    "y": 20
}

print("\nQuestion 8")

for key, value in data.items():
    print(key, ":", value)


# Question 9
data = {}

print("\nQuestion 9")
print(data.get("score", "Not Found"))


# Question 10
keys = ["a", "b"]
values = [1, 2]

dictionary = dict(zip(keys, values))

print("\nQuestion 10")
print(dictionary)


# End of Part D