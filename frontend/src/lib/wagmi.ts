
import { createConfig, http } from "wagmi";
import { optimism, sepolia } from "wagmi/chains";
import { injected } from "wagmi/connectors";

export const config = createConfig({
  chains: [optimism, sepolia],
  connectors: [
    injected(),
  ],
  transports: {
    [optimism.id]: http("https://mainnet.optimism.io"),
    [sepolia.id]: http(),
  },
});
