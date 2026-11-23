
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { network } from "hardhat";

describe("BudgetManager", async function () {
  const { viem } = await network.create();

  it("should set a Paymaster budget", async function () {
    const budgetManager = await viem.deployContract("BudgetManager");
    const [owner, paymaster] = await viem.getWalletClients();

    await budgetManager.write.setBudget([
      paymaster.account.address,
      1000n,
    ]);

    const budget = await budgetManager.read.budgets([
      paymaster.account.address,
    ]);

    assert.equal(budget[0], 1000n);
    assert.equal(budget[1], 0n);
    assert.equal(budget[2], 0n);
    assert.equal(budget[3], true);
  });

  it("should reserve sponsorship within the available budget", async function () {
    const budgetManager = await viem.deployContract("BudgetManager");
    const [owner, paymaster, user] = await viem.getWalletClients();

    await budgetManager.write.setBudget([
      paymaster.account.address,
      1000n,
    ]);

    await budgetManager.write.reserveSponsorship([
      paymaster.account.address,
      user.account.address,
      300n,
    ]);

    assert.equal(
      await budgetManager.read.availableBudget([
        paymaster.account.address,
      ]),
      700n,
    );
  });

  it("should settle a sponsorship reservation", async function () {
    const budgetManager = await viem.deployContract("BudgetManager");
    const [owner, paymaster, user] = await viem.getWalletClients();

    await budgetManager.write.setBudget([
      paymaster.account.address,
      1000n,
    ]);

    await budgetManager.write.reserveSponsorship([
      paymaster.account.address,
      user.account.address,
      300n,
    ]);

    await budgetManager.write.settleSponsorship([
      paymaster.account.address,
      user.account.address,
      300n,
      250n,
    ]);

    const budget = await budgetManager.read.budgets([
      paymaster.account.address,
    ]);

    assert.equal(budget[0], 1000n);
    assert.equal(budget[1], 250n);
    assert.equal(budget[2], 0n);
  });

  it("should reject reservations exceeding the available budget", async function () {
    const budgetManager = await viem.deployContract("BudgetManager");
    const [owner, paymaster, user] = await viem.getWalletClients();

    await budgetManager.write.setBudget([
      paymaster.account.address,
      100n,
    ]);

    await assert.rejects(
      budgetManager.write.reserveSponsorship([
        paymaster.account.address,
        user.account.address,
        150n,
      ]),
    );
  });

  it("should reject unauthorized budget changes", async function () {
    const budgetManager = await viem.deployContract("BudgetManager");
    const [owner, paymaster] = await viem.getWalletClients();

    const unauthorizedManager = await viem.getContractAt(
      "BudgetManager",
      budgetManager.address,
      { client: { wallet: paymaster } },
    );

    await assert.rejects(
      unauthorizedManager.write.setBudget([
        paymaster.account.address,
        1000n,
      ]),
    );
  });

  it("should reject reducing the budget below committed spending", async function () {
    const budgetManager = await viem.deployContract("BudgetManager");
    const [owner, paymaster, user] = await viem.getWalletClients();

    await budgetManager.write.setBudget([
      paymaster.account.address,
      1000n,
    ]);

    await budgetManager.write.reserveSponsorship([
      paymaster.account.address,
      user.account.address,
      300n,
    ]);

    await budgetManager.write.settleSponsorship([
      paymaster.account.address,
      user.account.address,
      300n,
      250n,
    ]);

    await assert.rejects(
      budgetManager.write.setBudget([
        paymaster.account.address,
        200n,
      ]),
    );
  });
});
