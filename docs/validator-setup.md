# Validator Setup Guide

> **DOCKER-BASED DEPLOYMENT**
> This guide covers Docker Compose deployment, the recommended way to run a validator.

## Overview

This guide walks you through:
1. Installing Docker
2. Creating and registering a Bittensor validator wallet
3. Configuring and starting your validator
4. Monitoring and updating it

For how the subnet works end to end, see [the system components in the miner guide](./miner-setup.md#system-components).

---

# What a validator does

A validator does not run anyone's code. It reads public data from the Numinous backend, checks it, scores every miner with a fixed formula and sets weights. Every validator computes the same scores from the same data, with no model involved.

| Task | Every | What it does |
|---|---|---|
| Sync miners | 5 min | Reads the subnet's current miners (uid and hotkey) from the chain |
| Pull rounds | 10 min | Stores every closed round, its questions and the hash of every forecast received before the deadline |
| Pull results | 15 min | Stores every opened forecast with the result of each judged statement, and checks each one against its hash and the miner's signature |
| Score miners | daily at 15:00 UTC | Computes every miner's credits from the statements judged correct in the last 30 days (see [scoring-system.md](./scoring-system.md)) |
| Post scores | 5 min | Sends each day's scores to the backend once |
| Set weights | ~6 min, within the chain's rate limit | Sets weights from the latest scores that are at least an hour old, so every validator switches at 16:00 UTC |
| Clean up | hourly | Keeps the last 90 days of rounds and everything under them, the same rule on every validator |

A forecast that does not match its hash or its miner's signature is stored with the reason and never counts.

---

# System Requirements

**Recommended Specifications:**
- **OS:** Ubuntu 22.04 LTS or Debian 11+
- **CPU:** 2 cores
- **RAM:** 4 GB
- **Disk:** 20 GB SSD (the database holds 90 days of rounds and stays small)
- **Network:** outbound HTTPS to the Numinous backend and a Subtensor endpoint. No inbound ports are needed.
- **Docker:** Docker Engine 20.10+ and Docker Compose V2

**Minimum Stake Requirements:**
- **Testnet (netuid 155):** Minimal TAO (see [wallet-setup.md](./wallet-setup.md#for-testnet) for the faucet)
- **Mainnet (netuid 6):** 10,000 TAO minimum stake

**Validator list:** the backend serves validator data only to the hotkeys on its validator list. Ask the Numinous team on [Discord](https://discord.gg/qKPeYPc3) to add your validator hotkey before you start.

---

# Setup Steps

## 1. Install Docker & Bittensor CLI

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y docker.io docker-compose-plugin git python3 python3-pip

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER

# Install Bittensor CLI
pip3 install bittensor

# Log out and back in for group changes to take effect
# Then verify installation
docker --version
docker compose version
btcli --version
```

**Note:** This guide uses Docker Compose V2 (`docker compose`).

## 2. Clone Repository

```bash
git clone https://github.com/numinouslabs/numinous.git
cd numinous
```

## 3. Setup Wallet

Create and register your validator wallet. See [wallet-setup.md](./wallet-setup.md) for detailed instructions.

```bash
# Create wallet
btcli wallet new-coldkey --wallet validator
btcli wallet new-hotkey --wallet validator --wallet-hotkey default

# Register on subnet
# Testnet: --netuid 155 --subtensor.network test
# Mainnet: --netuid 6 --subtensor.network finney
btcli subnet register --netuid 6 --wallet validator --wallet-hotkey default --network finney
```

## 4. Configure Environment Variables

Create the environment file from the template:

```bash
cp .env.validator.example .env.validator
```

Edit it with your wallet names:

```bash
# Must match the directory names under ~/.bittensor/wallets/
WALLET_NAME=validator
WALLET_HOTKEY=default
```

`docker-compose.validator.yaml` mounts `~/.bittensor` into the container. If your wallets live elsewhere, change the left side of that volume line to your wallet folder's parent.

Before starting, check the wallet files exist:

```bash
ls -la ~/.bittensor/wallets/${WALLET_NAME}/coldkeypub.txt
ls -la ~/.bittensor/wallets/${WALLET_NAME}/hotkeys/${WALLET_HOTKEY}
```

## 5. Start Validator

```bash
docker compose -f docker-compose.validator.yaml --env-file .env.validator pull
docker compose -f docker-compose.validator.yaml --env-file .env.validator up -d

# View logs
docker logs -f numinous_validator
```

On its first start the validator pulls the last 90 days of rounds and results, which can take a few minutes. It scores at the next 15:00 UTC, or right away when it starts after 15:00.

---

# Monitoring & Management

## Common Commands

```bash
# View logs
docker logs -f numinous_validator

# Check status
docker ps | grep numinous_validator

# Restart validator
docker compose -f docker-compose.validator.yaml --env-file .env.validator restart

# Stop validator
docker compose -f docker-compose.validator.yaml --env-file .env.validator down
```

## Updating

Always update the repository **before** pulling the new image: a new image can expect the command line of the new `docker-compose.validator.yaml`.

```bash
git pull
docker compose -f docker-compose.validator.yaml --env-file .env.validator pull
docker compose -f docker-compose.validator.yaml --env-file .env.validator up -d
```

## Healthy log lines

| Log message | Meaning |
|---|---|
| `Validator started` | Migrations ran and every task is scheduled |
| `DKG results pulled` | A results pull finished and was recorded |
| `DKG results waiting for their round, pull not recorded` | Some results arrived before their round was pulled; the next pull catches up |
| `DKG miners scored` | The daily scores were computed |
| `DKG scores posted` | The daily scores reached the backend |
| `Weights from DKG scores` / `Weights set successfully.` | Weights were set from the latest scores |
| `No DKG scores in effect yet, weights not set.` | Normal right after the first start, until the first scores are an hour old |

---

# Troubleshooting

```bash
# Check Docker is running
docker ps

# Check logs for errors
docker logs numinous_validator 2>&1 | grep -i error

# Verify the wallet configuration
cat .env.validator | grep WALLET
ls -la ~/.bittensor/wallets/validator/hotkeys/default
```

- **403 "this hotkey is not a validator of the subnet":** your hotkey is not on the backend's validator list yet; ask the team on Discord.
- **"unrecognized arguments" at start:** your `docker-compose.validator.yaml` is older than the image; run `git pull`, then `up -d` again.
- **The hotkey is not found:** check `WALLET_NAME`, `WALLET_HOTKEY` and the `~/.bittensor` volume line.
