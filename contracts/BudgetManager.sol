
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.34;

/// @title BudgetManager
/// @notice Manages sponsorship budgets and reservations for Paymasters.
/// @dev This is an accounting prototype, not an ERC-4337 Paymaster.
contract BudgetManager {
    address public owner;

    struct PaymasterBudget {
        uint256 totalBudget;
        uint256 spent;
        uint256 reserved;
        bool active;
    }

    mapping(address => PaymasterBudget) public budgets;

    event PaymasterBudgetUpdated(
        address indexed paymaster,
        uint256 totalBudget
    );

    event SponsorshipReserved(
        address indexed paymaster,
        address indexed user,
        uint256 amount
    );

    event SponsorshipSettled(
        address indexed paymaster,
        address indexed user,
        uint256 reservedAmount,
        uint256 actualCost
    );

    event PaymasterStatusUpdated(
        address indexed paymaster,
        bool active
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setBudget(
        address paymaster,
        uint256 totalBudget
    ) external onlyOwner {
        require(paymaster != address(0), "Invalid Paymaster");
        require(
            totalBudget >= budgets[paymaster].spent +
                budgets[paymaster].reserved,
            "Budget below committed amount"
        );

        budgets[paymaster].totalBudget = totalBudget;
        budgets[paymaster].active = true;

        emit PaymasterBudgetUpdated(paymaster, totalBudget);
    }

    function setPaymasterStatus(
        address paymaster,
        bool active
    ) external onlyOwner {
        require(paymaster != address(0), "Invalid Paymaster");

        budgets[paymaster].active = active;

        emit PaymasterStatusUpdated(paymaster, active);
    }

    function reserveSponsorship(
        address paymaster,
        address user,
        uint256 amount
    ) external onlyOwner {
        require(budgets[paymaster].active, "Paymaster inactive");
        require(user != address(0), "Invalid user");
        require(amount > 0, "Amount must be positive");

        PaymasterBudget storage budget = budgets[paymaster];

        require(
            amount <= budget.totalBudget - budget.spent - budget.reserved,
            "Insufficient budget"
        );

        budget.reserved += amount;

        emit SponsorshipReserved(paymaster, user, amount);
    }

    function settleSponsorship(
        address paymaster,
        address user,
        uint256 reservedAmount,
        uint256 actualCost
    ) external onlyOwner {
        PaymasterBudget storage budget = budgets[paymaster];

        require(reservedAmount > 0, "Invalid reservation");
        require(actualCost <= reservedAmount, "Actual cost exceeds reserve");
        require(
            budget.reserved >= reservedAmount,
            "Insufficient reserved amount"
        );

        budget.reserved -= reservedAmount;
        budget.spent += actualCost;

        emit SponsorshipSettled(
            paymaster,
            user,
            reservedAmount,
            actualCost
        );
    }

    function availableBudget(
        address paymaster
    ) external view returns (uint256) {
        PaymasterBudget storage budget = budgets[paymaster];

        return budget.totalBudget - budget.spent - budget.reserved;
    }
}
