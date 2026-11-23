
"use client";

import { useEffect, useState, type FormEvent } from "react";
import {
  useAccount,
  useConnect,
  useDisconnect,
  useChainId,
} from "wagmi";

const API_URL = "http://127.0.0.1:8000";

type DatasetSummary = {
  total_operations: number;
  unique_users: number;
  unique_paymasters: number;
  successful_operations: number;
  failed_operations: number;
  total_gas_spending_eth: number;
};

type AllocationDecision = {
  decision_order: number;
  candidate_index: number;
  sender: string;
  accepted: boolean;
  reason: string;
  predicted_gas_cost_eth: number;
  safety_multiplier: number;
  required_reserve_eth: number;
  reserved_gas_eth: number;
  budget_before_eth: number;
  budget_after_eth: number;
  fairness_score: number;
  spending_score: number;
  cost_score: number;
  proposed_score: number;
};

type AllocationResponse = {
  mode: string;
  warning: string;
  paymaster: string;
  strategy: string;
  prediction_method: string;
  prediction_history_size: number;
  predicted_cost_eth: number;
  safety_multiplier: number;
  required_reserve_per_operation_eth: number;
  initial_simulated_budget_eth: number;
  total_reserved_eth: number;
  remaining_simulated_budget_eth: number;
  accepted_count: number;
  rejected_count: number;
  total_candidates: number;
  decisions: AllocationDecision[];
};

const coverageData = [
  { safety: "1.00x", fcfs: 29.82, fairness: 29.85, proposed: 29.85 },
  { safety: "1.10x", fcfs: 28.82, fairness: 28.71, proposed: 28.71 },
  { safety: "1.25x", fcfs: 27.35, fairness: 27.35, proposed: 27.35 },
  { safety: "1.50x", fcfs: 25.17, fairness: 25.14, proposed: 25.14 },
  { safety: "2.00x", fcfs: 22.24, fairness: 22.21, proposed: 22.24 },
];

const fairnessData = [
  { safety: "1.00x", fcfs: 0.4850, fairness: 0.4875, proposed: 0.4875 },
  { safety: "1.10x", fcfs: 0.4916, fairness: 0.4876, proposed: 0.4876 },
  { safety: "1.25x", fcfs: 0.4968, fairness: 0.4969, proposed: 0.4969 },
  { safety: "1.50x", fcfs: 0.5242, fairness: 0.5425, proposed: 0.5425 },
  { safety: "2.00x", fcfs: 0.5962, fairness: 0.5989, proposed: 0.5985 },
];

const experimentRows = [
  { safety: "1.00x", fcfs: "1,160", fairness: "1,161", proposed: "1,161", excess: "0.0000013200" },
  { safety: "1.10x", fcfs: "1,121", fairness: "1,117", proposed: "1,117", excess: "0.0000009964" },
  { safety: "1.25x", fcfs: "1,064", fairness: "1,064", proposed: "1,064", excess: "0.0000003019" },
  { safety: "1.50x", fcfs: "979", fairness: "978", proposed: "978", excess: "0.0000000000" },
  { safety: "2.00x", fcfs: "865", fairness: "864", proposed: "865", excess: "0.0000000000" },
];

function SectionTitle({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <div className="mb-5">
      <h2 className="text-lg font-semibold text-white">{title}</h2>
      <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  color,
}: {
  label: string;
  value: string;
  detail: string;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className={`mt-3 text-3xl font-bold ${color}`}>{value}</p>
      <p className="mt-2 text-xs text-slate-500">{detail}</p>
    </div>
  );
}

function CoverageChart() {
  return (
    <div className="space-y-5">
      {coverageData.map((item) => (
        <div key={item.safety}>
          <p className="mb-2 text-xs font-medium text-slate-300">
            Safety {item.safety}
          </p>

          {[
            { name: "FCFS", value: item.fcfs, color: "bg-blue-500" },
            { name: "Fairness", value: item.fairness, color: "bg-cyan-400" },
            { name: "Proposed", value: item.proposed, color: "bg-violet-400" },
          ].map((bar) => (
            <div key={bar.name} className="mb-2 flex items-center gap-3">
              <span className="w-16 text-xs text-slate-500">{bar.name}</span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full ${bar.color}`}
                  style={{ width: `${(bar.value / 40) * 100}%` }}
                />
              </div>
              <span className="w-14 text-right text-xs text-slate-300">
                {bar.value.toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function FairnessChart() {
  return (
    <div className="space-y-5">
      {fairnessData.map((item) => (
        <div key={item.safety}>
          <p className="mb-2 text-xs font-medium text-slate-300">
            Safety {item.safety}
          </p>

          {[
            { name: "FCFS", value: item.fcfs, color: "bg-blue-500" },
            { name: "Fairness", value: item.fairness, color: "bg-cyan-400" },
            { name: "Proposed", value: item.proposed, color: "bg-violet-400" },
          ].map((bar) => (
            <div key={bar.name} className="mb-2 flex items-center gap-3">
              <span className="w-16 text-xs text-slate-500">{bar.name}</span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full ${bar.color}`}
                  style={{ width: `${bar.value * 100}%` }}
                />
              </div>
              <span className="w-14 text-right text-xs text-slate-300">
                {bar.value.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  // Prevent MetaMask browser state from causing a server/client hydration mismatch.
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // MetaMask wallet connection (wallet identity only; no transaction is sent here).
  const { address, isConnected } = useAccount();

  const {
    connect,
    connectors,
    isPending,
    error: connectError,
  } = useConnect();

  const { disconnect } = useDisconnect();
  const chainId = useChainId();

  const injectedConnector = connectors.find(
    (connector) => connector.id === "injected"
  );

  const networkName =
    chainId === 10
      ? "Optimism Mainnet"
      : chainId === 11155111
        ? "Sepolia Testnet"
        : `Other network (Chain ID: ${chainId})`;

  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [apiError, setApiError] = useState("");

  const [paymaster, setPaymaster] = useState(
    "0x0000000000000000000000000000000000000000"
  );

  const [senders, setSenders] = useState(
    "0x1111111111111111111111111111111111111111\n0x2222222222222222222222222222222222222222\n0x3333333333333333333333333333333333333333"
  );

  const [budget, setBudget] = useState("0.00001");
  const [strategy, setStrategy] = useState("PROPOSED");
  const [safety, setSafety] = useState("1.5");

  const [allocation, setAllocation] = useState<AllocationResponse | null>(null);
  const [allocationError, setAllocationError] = useState("");
  const [allocationLoading, setAllocationLoading] = useState(false);

  useEffect(() => {
    async function loadDataset() {
      try {
        const response = await fetch(`${API_URL}/dataset/summary`);

        if (!response.ok) {
          throw new Error("Backend API request failed");
        }

        const data: DatasetSummary = await response.json();
        setDataset(data);
        setApiError("");
      } catch (error) {
        console.error("Dataset API error:", error);
        setApiError(
          "Could not connect to the backend. Check that FastAPI is running."
        );
      }
    }

    loadDataset();
  }, []);

  async function runAllocation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setAllocationError("");
    setAllocation(null);

    const candidateAddresses = senders
      .split(/[\n,]+/)
      .map((address) => address.trim())
      .filter(Boolean);

    const addressPattern = /^0x[a-fA-F0-9]{40}$/;

    if (!addressPattern.test(paymaster.trim())) {
      setAllocationError("Enter a valid Paymaster Ethereum address.");
      return;
    }

    if (candidateAddresses.length === 0) {
      setAllocationError("Enter at least one candidate wallet address.");
      return;
    }

    if (candidateAddresses.length > 100) {
      setAllocationError("Maximum 100 candidate addresses are allowed.");
      return;
    }

    if (candidateAddresses.some((address) => !addressPattern.test(address))) {
      setAllocationError("One or more candidate wallet addresses are invalid.");
      return;
    }

    const budgetValue = Number(budget);
    const safetyValue = Number(safety);

    if (
      !Number.isFinite(budgetValue) ||
      budgetValue < 0 ||
      budgetValue > 1000
    ) {
      setAllocationError("Budget must be between 0 and 1000 ETH.");
      return;
    }

    if (
      !Number.isFinite(safetyValue) ||
      safetyValue < 1 ||
      safetyValue > 5
    ) {
      setAllocationError("Safety multiplier must be between 1 and 5.");
      return;
    }

    setAllocationLoading(true);

    try {
      const response = await fetch(`${API_URL}/allocation/decide`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          paymaster: paymaster.trim(),
          candidates: candidateAddresses.map((sender) => ({ sender })),
          simulated_budget_eth: budgetValue,
          strategy,
          safety_multiplier: safetyValue,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail ?? "Allocation request failed.")
        );
      }

      setAllocation(data as AllocationResponse);
    } catch (error) {
      setAllocationError(
        error instanceof Error ? error.message : "Unexpected API error."
      );
    } finally {
      setAllocationLoading(false);
    }
  }

  const stats = [
    {
      label: "UserOperations",
      value: dataset ? dataset.total_operations.toLocaleString() : "Loading...",
      detail: "Observed on Optimism",
      color: "text-blue-400",
    },
    {
      label: "Unique Users",
      value: dataset ? dataset.unique_users.toLocaleString() : "Loading...",
      detail: "Distinct smart-account senders",
      color: "text-cyan-400",
    },
    {
      label: "Paymasters",
      value: dataset ? dataset.unique_paymasters.toString() : "Loading...",
      detail: "Observed in the dataset",
      color: "text-violet-400",
    },
    {
      label: "Success Rate",
      value:
        dataset && dataset.total_operations > 0
          ? `${(
              (dataset.successful_operations / dataset.total_operations) *
              100
            ).toFixed(2)}%`
          : "Loading...",
      detail: dataset
        ? `${dataset.successful_operations.toLocaleString()} successful operations`
        : "Fetching dataset...",
      color: "text-emerald-400",
    },
  ];

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8">

        <header className="mb-9 flex flex-col justify-between gap-5 border-b border-slate-800 pb-7 sm:flex-row sm:items-center">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
              <span className="text-xs font-medium uppercase tracking-widest text-emerald-400">
                Research Analytics Platform
              </span>
            </div>

            <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
              Gas Sponsorship Dashboard
            </h1>

            <p className="mt-2 text-sm text-slate-400">
              Budget-Aware Gas Sponsorship Allocation for ERC-4337
              Smart-Account Ecosystems
            </p>
          </div>

          <div className="min-w-64 rounded-lg border border-slate-800 bg-slate-900 px-4 py-3">
            <p className="text-xs text-slate-500">MetaMask Wallet</p>

            {!mounted ? (
              <p className="mt-2 text-sm text-slate-400">
                Checking wallet...
              </p>
            ) : isConnected ? (
              <>
                <p className="mt-2 text-sm font-semibold text-emerald-400">
                  Wallet Connected
                </p>
                <p className="mt-2 break-all font-mono text-xs text-white">
                  {address}
                </p>
                <p className="mt-2 text-xs text-cyan-400">{networkName}</p>
                <p className="mt-1 text-xs text-slate-500">
                  Chain ID: {chainId}
                </p>
                <button
                  type="button"
                  onClick={() => disconnect()}
                  className="mt-3 rounded-lg bg-red-600 px-3 py-2 text-xs font-semibold text-white hover:bg-red-500"
                >
                  Disconnect Wallet
                </button>
              </>
            ) : (
              <>
                <p className="mt-2 text-sm text-slate-400">
                  Wallet Not Connected
                </p>
                <button
                  type="button"
                  onClick={() => {
                    if (injectedConnector) {
                      connect({ connector: injectedConnector });
                    }
                  }}
                  disabled={!injectedConnector || isPending}
                  className="mt-3 rounded-lg bg-orange-500 px-4 py-2 text-sm font-semibold text-white hover:bg-orange-400 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isPending ? "Connecting..." : "Connect MetaMask"}
                </button>
                {!injectedConnector && (
                  <p className="mt-2 text-xs text-amber-400">
                    No browser wallet detected. Install or enable MetaMask, then refresh.
                  </p>
                )}
                {connectError && (
                  <p className="mt-2 max-w-xs break-words text-xs text-red-400">
                    {connectError.message}
                  </p>
                )}
              </>
            )}
          </div>
        </header>

        <div className="mb-6 rounded-lg border border-slate-800 bg-slate-900 px-4 py-3">
          {apiError ? (
            <p className="text-sm text-red-400">{apiError}</p>
          ) : dataset ? (
            <p className="text-sm text-emerald-400">
              ● Backend connected · Live dataset statistics loaded
            </p>
          ) : (
            <p className="text-sm text-amber-400">
              ● Connecting to backend...
            </p>
          )}
        </div>

        <section className="mb-10">
          <SectionTitle
            title="Dataset Overview"
            subtitle="Observed ERC-4337 UserOperationEvent data from Optimism"
          />

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {stats.map((stat) => (
              <MetricCard key={stat.label} {...stat} />
            ))}
          </div>
        </section>

        <section className="mb-10 grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-sm text-slate-400">
              Total observed gas spending
            </p>
            <p className="mt-3 text-2xl font-bold text-white">
              {dataset
                ? `${dataset.total_gas_spending_eth.toFixed(8)} ETH`
                : "Loading..."}
            </p>
            <p className="mt-2 text-xs text-slate-500">
              Across all collected operations
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-sm text-slate-400">
              Successful operations
            </p>
            <p className="mt-3 text-2xl font-bold text-emerald-400">
              {dataset
                ? dataset.successful_operations.toLocaleString()
                : "Loading..."}
            </p>
            <p className="mt-2 text-xs text-slate-500">
              {dataset
                ? `${dataset.failed_operations} unsuccessful operations`
                : "Fetching data..."}
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-sm text-slate-400">Global validation</p>
            <p className="mt-3 text-2xl font-bold text-cyan-400">
              45 experiments
            </p>
            <p className="mt-2 text-xs text-slate-500">
              3 strategies × 3 budgets × 5 safety margins
            </p>
          </div>
        </section>

        {/* Sponsorship allocation simulator */}
        <section className="mb-10">
          <SectionTitle
            title="Sponsorship Allocation Simulator"
            subtitle="Submit candidate smart accounts to the FastAPI allocation endpoint"
          />

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <form
              onSubmit={runAllocation}
              className="space-y-5 rounded-xl border border-slate-800 bg-slate-900 p-6"
            >
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-300">
                  Paymaster Ethereum address
                </label>

                <input
                  value={paymaster}
                  onChange={(event) => setPaymaster(event.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 font-mono text-xs text-white outline-none focus:border-blue-500"
                  placeholder="0x..."
                  required
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-slate-300">
                  Candidate wallet addresses
                </label>

                <textarea
                  value={senders}
                  onChange={(event) => setSenders(event.target.value)}
                  rows={5}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 font-mono text-xs text-white outline-none focus:border-blue-500"
                  placeholder="Enter one wallet address per line"
                  required
                />

                <p className="mt-1 text-xs text-slate-500">
                  Enter one address per line. Maximum 100 candidates.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <label className="mb-2 block text-sm text-slate-300">
                    Simulated budget (ETH)
                  </label>

                  <input
                    type="number"
                    min="0"
                    max="1000"
                    step="any"
                    value={budget}
                    onChange={(event) => setBudget(event.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white outline-none focus:border-blue-500"
                    required
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm text-slate-300">
                    Strategy
                  </label>

                  <select
                    value={strategy}
                    onChange={(event) => setStrategy(event.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white outline-none focus:border-blue-500"
                  >
                    <option value="FCFS">FCFS</option>
                    <option value="FAIRNESS">Fairness</option>
                    <option value="PROPOSED">Proposed</option>
                  </select>
                </div>

                <div>
                  <label className="mb-2 block text-sm text-slate-300">
                    Safety multiplier
                  </label>

                  <input
                    type="number"
                    min="1"
                    max="5"
                    step="0.1"
                    value={safety}
                    onChange={(event) => setSafety(event.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white outline-none focus:border-blue-500"
                    required
                  />
                </div>
              </div>

              {allocationError && (
                <div className="rounded-lg border border-red-900 bg-red-950/40 p-3 text-sm text-red-300">
                  {allocationError}
                </div>
              )}

              <button
                type="submit"
                disabled={allocationLoading}
                className="w-full rounded-lg bg-blue-600 px-5 py-3 font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {allocationLoading
                  ? "Calculating allocation..."
                  : "Run Sponsorship Allocation"}
              </button>

              <p className="text-xs leading-5 text-amber-400">
                Simulation only. No actual Paymaster balances are accessed and
                no blockchain transactions are sent.
              </p>
            </form>

            {/* Allocation results */}
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="mb-4 font-semibold text-white">
                Allocation Results
              </h3>

              {!allocation && !allocationLoading && (
                <div className="flex min-h-64 items-center justify-center rounded-lg border border-dashed border-slate-700 p-6 text-center text-sm text-slate-500">
                  Submit the form to see sponsorship decisions here.
                </div>
              )}

              {allocationLoading && (
                <div className="flex min-h-64 items-center justify-center text-sm text-blue-400">
                  Processing your allocation request...
                </div>
              )}

              {allocation && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard
                      label="Accepted"
                      value={String(allocation.accepted_count)}
                      detail="Candidates accepted"
                      color="text-emerald-400"
                    />

                    <MetricCard
                      label="Rejected"
                      value={String(allocation.rejected_count)}
                      detail="Candidates rejected"
                      color="text-red-400"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-slate-950 p-3">
                      <p className="text-xs text-slate-500">Predicted cost</p>
                      <p className="mt-2 break-all font-mono text-sm text-white">
                        {allocation.predicted_cost_eth.toExponential(5)} ETH
                      </p>
                    </div>

                    <div className="rounded-lg bg-slate-950 p-3">
                      <p className="text-xs text-slate-500">Total reserved</p>
                      <p className="mt-2 break-all font-mono text-sm text-white">
                        {allocation.total_reserved_eth.toExponential(5)} ETH
                      </p>
                    </div>

                    <div className="rounded-lg bg-slate-950 p-3">
                      <p className="text-xs text-slate-500">
                        Remaining budget
                      </p>
                      <p className="mt-2 break-all font-mono text-sm text-emerald-400">
                        {allocation.remaining_simulated_budget_eth.toExponential(5)} ETH
                      </p>
                    </div>

                    <div className="rounded-lg bg-slate-950 p-3">
                      <p className="text-xs text-slate-500">
                        Prediction method
                      </p>
                      <p className="mt-2 break-words text-sm text-cyan-400">
                        {allocation.prediction_method}
                      </p>
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-700 p-3">
                    <p className="text-xs text-slate-500">Paymaster</p>
                    <p className="mt-1 break-all font-mono text-xs text-slate-300">
                      {allocation.paymaster}
                    </p>

                    <p className="mt-3 text-xs text-slate-500">
                      Strategy: {allocation.strategy} · Safety:{" "}
                      {allocation.safety_multiplier}×
                    </p>

                    <p className="mt-1 text-xs text-slate-500">
                      Prediction history: {allocation.prediction_history_size}{" "}
                      operations
                    </p>
                  </div>

                  <h4 className="font-semibold text-white">
                    Individual decisions
                  </h4>

                  <div className="max-h-80 space-y-3 overflow-y-auto">
                    {allocation.decisions.map((decision) => (
                      <div
                        key={decision.candidate_index}
                        className="rounded-lg border border-slate-700 bg-slate-950 p-3"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-sm font-medium text-white">
                            Candidate {decision.decision_order}
                          </span>

                          <span
                            className={
                              decision.accepted
                                ? "rounded-full bg-emerald-950 px-2 py-1 text-xs text-emerald-400"
                                : "rounded-full bg-red-950 px-2 py-1 text-xs text-red-400"
                            }
                          >
                            {decision.accepted ? "Accepted" : "Rejected"}
                          </span>
                        </div>

                        <p className="mt-2 break-all font-mono text-xs text-slate-400">
                          {decision.sender}
                        </p>

                        <p className="mt-2 text-xs text-slate-300">
                          {decision.reason}
                        </p>

                        <p className="mt-2 text-xs text-slate-500">
                          Reserved:{" "}
                          {decision.reserved_gas_eth.toExponential(5)} ETH
                        </p>
                      </div>
                    ))}
                  </div>

                  <p className="rounded-lg border border-amber-900 bg-amber-950/30 p-3 text-xs leading-5 text-amber-300">
                    {allocation.warning}
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="mb-10">
          <SectionTitle
            title="Allocation Experiment Analysis"
            subtitle="Global chronological validation · Simulated budget multiplier: 2.5x"
          />

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <h3 className="mb-5 font-semibold text-white">
                Operation Coverage
              </h3>
              <CoverageChart />
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <h3 className="mb-5 font-semibold text-white">
                Global Jain Fairness Index
              </h3>
              <FairnessChart />
            </div>
          </div>
        </section>

        <section className="mb-10">
          <SectionTitle
            title="Strategy Comparison"
            subtitle="Global chronological validation · Budget multiplier 2.5x"
          />

          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full min-w-[700px] text-left text-sm">
              <thead className="bg-slate-900 text-xs uppercase text-slate-400">
                <tr>
                  <th className="px-5 py-4">Safety</th>
                  <th className="px-5 py-4">FCFS</th>
                  <th className="px-5 py-4">Fairness</th>
                  <th className="px-5 py-4">Proposed</th>
                  <th className="px-5 py-4">Budget excess (ETH)</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-800 bg-slate-950">
                {experimentRows.map((row) => (
                  <tr key={row.safety}>
                    <td className="px-5 py-4 font-medium text-white">
                      {row.safety}
                    </td>
                    <td className="px-5 py-4 text-blue-400">{row.fcfs}</td>
                    <td className="px-5 py-4 text-cyan-400">
                      {row.fairness}
                    </td>
                    <td className="px-5 py-4 text-violet-400">
                      {row.proposed}
                    </td>
                    <td className="px-5 py-4 font-mono text-xs text-slate-400">
                      {row.excess}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="mb-10 rounded-xl border border-slate-800 bg-slate-900 p-6">
          <h2 className="text-lg font-semibold text-white">
            Research Observations
          </h2>

          <ul className="mt-4 list-disc space-y-3 pl-5 text-sm leading-6 text-slate-400">
            <li>
              Higher safety margins reduce budget overruns but also reduce
              sponsorship coverage.
            </li>
            <li>
              Fairness-based allocation can improve the distribution of
              sponsorship among users.
            </li>
            <li>
              The proposed weighted strategy does not consistently outperform
              fairness-only allocation.
            </li>
            <li>
              These results are historical simulations using simulated budgets,
              not actual Paymaster balances.
            </li>
          </ul>
        </section>

        <footer className="border-t border-slate-800 py-6 text-center">
          <p className="text-xs text-slate-500">
            ERC-4337 Budget-Aware Gas Sponsorship Research Project
          </p>
          <p className="mt-2 text-xs text-slate-600">
            Experimental dashboard · Historical Optimism blockchain data
          </p>
        </footer>
      </div>
    </main>
  );
}
