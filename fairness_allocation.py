
from decimal import Decimal

# Sponsor's available budget
budget = Decimal("0.50")

# User information
users = [
    {"name": "User A", "previous_sponsorships": 10, "request": Decimal("0.20")},
    {"name": "User B", "previous_sponsorships": 1, "request": Decimal("0.20")},
    {"name": "User C", "previous_sponsorships": 5, "request": Decimal("0.15")},
    {"name": "User D", "previous_sponsorships": 2, "request": Decimal("0.10")}
]

# Give priority to users with fewer previous sponsorships
users.sort(key=lambda user: user["previous_sponsorships"])

print("FAIR GAS SPONSORSHIP SYSTEM")
print("---------------------------")

for user in users:

    print("\nUser:", user["name"])
    print("Previous sponsorships:", user["previous_sponsorships"])
    print("Requested:", user["request"], "ETH")

    if user["request"] <= budget:
        budget -= user["request"]
        print("Status: APPROVED")
    else:
        print("Status: REJECTED - Insufficient budget")

    print("Remaining budget:", budget, "ETH")

print("\nFinal remaining budget:", budget, "ETH")
