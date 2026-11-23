
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { network } from "hardhat";

describe("BudgetPaymaster", async function () {
  const { viem } = await network.connect();

  it("should deploy with the correct EntryPoint and authorization signer", async function () {
    const publicClient = await viem.getPublicClient();
    const [, signer] = await viem.getWalletClients();

    // Deploy a local mock EntryPoint.
    const entryPoint = await viem.deployContract("MockEntryPoint");

    // Deploy BudgetPaymaster.
    const paymaster = await viem.deployContract("BudgetPaymaster", [
      entryPoint.address,
      signer.account.address,
    ]);

    // Verify the EntryPoint address.
    const storedEntryPoint = await paymaster.read.entryPoint();

    assert.equal(
      storedEntryPoint.toLowerCase(),
      entryPoint.address.toLowerCase()
    );

    // Verify the authorization signer.
    const storedSigner = await paymaster.read.authorizationSigner();

    assert.equal(
      storedSigner.toLowerCase(),
      signer.account.address.toLowerCase()
    );

    // Verify deployed bytecode.
    const code = await publicClient.getCode({
      address: paymaster.address,
    });

    assert.ok(code && code !== "0x");
  });
});
