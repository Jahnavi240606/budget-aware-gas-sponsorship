
# Jain's Fairness Index

# Sponsorship amounts allocated to each user
allocations = [0.00, 0.20, 0.15, 0.10]

# Number of users
n = len(allocations)

# Calculate sum of allocations
total = sum(allocations)

# Calculate sum of squared allocations
sum_squares = sum(x ** 2 for x in allocations)

# Calculate Jain's Fairness Index
if sum_squares == 0:
    fairness_index = 0
else:
    fairness_index = (total ** 2) / (n * sum_squares)

print("FAIRNESS EVALUATION")
print("-------------------")

print("Total users:", n)
print("Total sponsorship allocated:", total, "ETH")

print("Jain's Fairness Index:", round(fairness_index, 4))

if fairness_index >= 0.9:
    print("Fairness: Very high")
elif fairness_index >= 0.7:
    print("Fairness: Moderate")
else:
    print("Fairness: Low")
