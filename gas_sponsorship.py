
from decimal import Decimal

# Sponsor's total budget
total_budget = Decimal("1.00")

# Sponsorship requests
requests = [
    Decimal("0.20"),
    Decimal("0.30"),
    Decimal("0.15"),
    Decimal("0.40"),
    Decimal("0.25")
]

remaining_budget = total_budget

print("GAS SPONSORSHIP SYSTEM")
print("----------------------")

for i, cost in enumerate(requests, start=1):

    print(f"\nUser {i} requested {cost} ETH")

    if cost <= remaining_budget:
        remaining_budget -= cost
        print("Status: APPROVED")
    else:
        print("Status: REJECTED - Insufficient budget")

    print(f"Remaining budget: {remaining_budget:.2f} ETH")

print("\nFinal remaining budget:", remaining_budget, "ETH")
