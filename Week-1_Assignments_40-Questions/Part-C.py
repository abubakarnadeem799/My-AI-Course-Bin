# Part C - Python Sets

# Question 1
numbers = [1, 2, 2, 3]
unique_numbers = set(numbers)

print("Question 1")
print(unique_numbers)


# Question 2
set1 = {1, 2, 3}
set1.add(4)

print("\nQuestion 2")
print(set1)


# Question 3
set1 = {1, 2, 3}
set1.remove(2)

print("\nQuestion 3")
print(set1)


# Question 4
set1 = {1, 3, 5}

print("\nQuestion 4")
print(5 in set1)


# Question 5
set1 = {10, 20, 30}

print("\nQuestion 5")
print("Length of Set:", len(set1))


# Question 6
set1 = {1, 2, 3}
set1.clear()

print("\nQuestion 6")
print(set1)


# Question 7
letters = {"a", "b"}

if "c" not in letters:
    letters.add("c")

print("\nQuestion 7")
print(letters)


# Question 8
letters = ["a", "a", "b"]
unique_letters = set(letters)

print("\nQuestion 8")
print(unique_letters)


# Question 9
set1 = {1, 2, 3}
set2 = {3, 4, 5}

print("\nQuestion 9")
print("Union:", set1 | set2)


# Question 10
set1 = {1, 2, 3}
set2 = {2, 3, 4}

print("\nQuestion 10")
print("Intersection:", set1 & set2)


# End of Part C