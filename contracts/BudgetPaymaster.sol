
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.34;

import "@account-abstraction/contracts/core/BasePaymaster.sol";
import "@account-abstraction/contracts/interfaces/PackedUserOperation.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

/// @title BudgetPaymaster
/// @notice ERC-4337 v0.7 Paymaster with signed sponsorship authorization.
/// @dev Research prototype. Not production-ready.
contract BudgetPaymaster is BasePaymaster {
    address public authorizationSigner;

    mapping(address => uint256) public spendingLimit;
    mapping(address => uint256) public spent;
    mapping(bytes32 => bool) public usedAuthorizations;

    event AuthorizationSignerUpdated(address indexed newSigner);

    event SpendingLimitUpdated(
        address indexed account,
        uint256 limit
    );

    event AuthorizationConsumed(
        address indexed account,
        bytes32 indexed digest
    );

    event SponsorshipCharged(
        address indexed account,
        uint256 actualCost
    );

    constructor(
        IEntryPoint entryPoint_,
        address signer_
    ) BasePaymaster(entryPoint_) {
        require(signer_ != address(0), "Invalid signer");
        authorizationSigner = signer_;
    }

    // --------------------------------------------------
    // ADMINISTRATION
    // --------------------------------------------------

    function setAuthorizationSigner(
        address newSigner
    ) external onlyOwner {
        require(newSigner != address(0), "Invalid signer");

        authorizationSigner = newSigner;

        emit AuthorizationSignerUpdated(newSigner);
    }

    function setSpendingLimit(
        address account,
        uint256 limit
    ) external onlyOwner {
        require(account != address(0), "Invalid account");
        require(limit >= spent[account], "Limit below spent");

        spendingLimit[account] = limit;

        emit SpendingLimitUpdated(account, limit);
    }

    // --------------------------------------------------
    // ERC-4337 VALIDATION
    // --------------------------------------------------

    function _validatePaymasterUserOp(
        PackedUserOperation calldata userOp,
        bytes32 userOpHash,
        uint256 maxCost
    )
        internal
        view
        override
        returns (bytes memory context, uint256 validationData)
    {
        bytes calldata data = userOp.paymasterAndData;

        // Expected layout:
        // 0-19    Paymaster address
        // 20-35   Verification gas limit
        // 36-51   Post-operation gas limit
        // 52-83   Expiry
        // 84-115  Authorized maximum cost
        // 116-147 Authorization nonce
        // 148-179 Authorization digest
        // 180-244 Signature

        require(
            data.length == 245,
            "Invalid paymaster data length"
        );

        uint48 expiry;
        uint256 authorizedMaxCost;
        uint256 authorizationNonce;
        bytes32 authorizationDigest;

        assembly {
            expiry := shr(208, calldataload(add(data.offset, 52)))
            authorizedMaxCost := calldataload(add(data.offset, 84))
            authorizationNonce := calldataload(add(data.offset, 116))
            authorizationDigest := calldataload(add(data.offset, 148))
        }

        bytes memory signature = data[180:245];

        require(
            block.timestamp <= expiry,
            "Authorization expired"
        );

        require(
            maxCost <= authorizedMaxCost,
            "Cost exceeds authorization"
        );

        uint256 limit = spendingLimit[userOp.sender];
        uint256 alreadySpent = spent[userOp.sender];

        require(
            alreadySpent <= limit,
            "Spending limit exceeded"
        );

        require(
            maxCost <= limit - alreadySpent,
            "Insufficient spending limit"
        );

        // Calculate the non-circular authorization digest.
        bytes32 expectedDigest = _computeAuthorizationDigest(
            userOp,
            authorizedMaxCost,
            expiry,
            authorizationNonce,
            keccak256(data[:52])
        );

        require(
            authorizationDigest == expectedDigest,
            "Invalid authorization digest"
        );

        require(
            !usedAuthorizations[authorizationDigest],
            "Authorization already used"
        );

        // Verify the off-chain authorization signature.
        bytes32 signedDigest =
            MessageHashUtils.toEthSignedMessageHash(expectedDigest);

        address recoveredSigner = ECDSA.recover(
            signedDigest,
            signature
        );

        require(
            recoveredSigner == authorizationSigner,
            "Invalid authorization signer"
        );

        context = abi.encode(
            userOp.sender,
            userOpHash,
            authorizationDigest,
            authorizedMaxCost
        );

        validationData = 0;
    }

    // --------------------------------------------------
    // ERC-4337 POST-OPERATION ACCOUNTING
    // --------------------------------------------------

    function _postOp(
        PostOpMode,
        bytes calldata context,
        uint256 actualGasCost,
        uint256
    ) internal override {
        (
            address account,
            ,
            bytes32 authorizationDigest,
            uint256 authorizedMaxCost
        ) = abi.decode(
            context,
            (address, bytes32, bytes32, uint256)
        );

        require(
            !usedAuthorizations[authorizationDigest],
            "Authorization already consumed"
        );

        require(
            actualGasCost <= authorizedMaxCost,
            "Actual cost exceeds authorization"
        );

        uint256 limit = spendingLimit[account];
        uint256 alreadySpent = spent[account];

        require(
            alreadySpent <= limit,
            "Spending limit exceeded"
        );

        require(
            actualGasCost <= limit - alreadySpent,
            "Insufficient limit"
        );

        usedAuthorizations[authorizationDigest] = true;

        spent[account] = alreadySpent + actualGasCost;

        emit AuthorizationConsumed(
            account,
            authorizationDigest
        );

        emit SponsorshipCharged(
            account,
            actualGasCost
        );
    }

    // --------------------------------------------------
    // AUTHORIZATION DIGEST
    // --------------------------------------------------

    function _computeAuthorizationDigest(
        PackedUserOperation calldata userOp,
        uint256 authorizedMaxCost,
        uint48 expiry,
        uint256 authorizationNonce,
        bytes32 paymasterFieldsHash
    ) internal view returns (bytes32) {
        return keccak256(
            abi.encode(
                block.chainid,
                address(this),
                userOp.sender,
                userOp.nonce,
                keccak256(userOp.initCode),
                keccak256(userOp.callData),
                userOp.accountGasLimits,
                userOp.preVerificationGas,
                userOp.gasFees,
                paymasterFieldsHash,
                authorizedMaxCost,
                expiry,
                authorizationNonce
            )
        );
    }
}
