# Validator Setup Guide

> **DOCKER-BASED DEPLOYMENT**
> This guide covers Docker Compose deployment - the recommended method for running validators.
> Docker ensures consistent environments, simplified updates, and production-ready isolation.

## Overview

This guide walks you through:
1. Installing Docker and system dependencies
2. Creating and registering a Bittensor validator wallet
3. Configuring your validator environment
4. Starting and verifying your validator
5. Monitoring and maintaining your validator

For system architecture details, see [architecture.md](./architecture.md).

---

# System Requirements

**Recommended Specifications:**
- **OS:** Ubuntu 22.04 LTS or Debian 11+
- **RAM:** 16 GB minimum (32 GB recommended for concurrent sandbox execution)
- **CPU:** 4 cores minimum (8+ recommended)
- **Disk:** 100 GB SSD with good IOPS (database + agent storage + sandbox space)
- **Network:** Stable internet connection with low latency
- **Docker:** Docker Engine 20.10+ and Docker Compose V2

**Minimum Stake Requirements:**
- **Testnet (netuid 155):** Minimal TAO (see [wallet-setup.md](./wallet-setup.md#for-testnet) for the faucet)
- **Mainnet (netuid 6):** 10,000 TAO minimum stake

---

# Architecture Overview

**How Validators Work:**

1. **Pull Events:** Fetch binary prediction events from Numinous API
2. **Pull Agent Code:** Download Python agent code submitted by miners via API
3. **Execute in Sandboxes:** Run each miner's agent in isolated Docker containers
4. **Collect Predictions:** Store predictions with run_id (execution UUID) and version_id (agent version)
5. **Report Forecasts:** Send forecasts and updated agent memory to the backend, which computes the scores
6. **Set Weights:** Update Bittensor subnet weights based on scores

**Sandbox Architecture:**
- Validators run in a container but create **sibling** sandbox containers
- Sandbox containers have NO internet access (isolated network)
- Agents make API calls via signing proxy (validator authenticates on their behalf)
- Each execution is isolated with unique temp directory and strict resource limits

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
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default

# Register on subnet
# Testnet: --netuid 155 --subtensor.network test
# Mainnet: --netuid 6 --subtensor.network finney
btcli subnet register --wallet.name validator --wallet.hotkey default --netuid 6 --subtensor.network finney
```

## 4. Configure Environment Variables

Create environment file from template:

```bash
cp .env.validator.example .env.validator
```

Edit the file with your wallet configuration:

```bash
vim .env.validator
```

**Required variables:**

```bash
# Wallet Configuration
# Must match directory names under ~/.bittensor/wallets/
WALLET_NAME=validator
WALLET_HOTKEY=default
```

**Save and exit**

## 5. Understanding Wallet Paths (Important)

The validator runs inside a Docker container but needs to access your wallet files from the **host machine**. The `HOST_WALLET_PATH` environment variable tells the validator where to find wallets on your host filesystem.

### Default Configuration

The `docker-compose.validator.yaml` file is pre-configured with:

```yaml
environment:
  - HOST_WALLET_PATH=${HOME}/.bittensor/wallets
```

This works for **most users** who store wallets in the standard Bittensor location (`~/.bittensor/wallets`).

When you run docker compose:
- `${HOME}` automatically expands to your home directory
- For user `bob`: `/home/bob/.bittensor/wallets`
- For user `root`: `/root/.bittensor/wallets`

### Custom Wallet Location

If you store wallets in a non-standard location, edit your `.env.validator` file:

```bash
# Add or change this line in .env.validator
HOST_WALLET_PATH=/your/custom/path/to/wallets
```

Replace `/your/custom/path/to/wallets` with your actual wallet directory path on the host machine.

### Verification

Before starting, verify your wallet files exist at the expected location:

```bash
# Check wallet structure
ls -la ~/.bittensor/wallets/${WALLET_NAME}/coldkey
ls -la ~/.bittensor/wallets/${WALLET_NAME}/hotkeys/${WALLET_HOTKEY}

# If files exist, you're ready to start!
```

### Troubleshooting

If you see errors about wallet files not found:

1. **Verify wallet path:** Check that `HOST_WALLET_PATH` matches where your wallets actually are
2. **Check permissions:** Ensure wallet files are readable
3. **Restart validator:** After changing `HOST_WALLET_PATH`, restart the validator

## 6. Start Validator

```bash
# Pull and start validator
docker compose -f docker-compose.validator.yaml --env-file .env.validator pull
docker compose -f docker-compose.validator.yaml --env-file .env.validator up -d

# View logs
docker logs -f numinous_validator
```

# Monitoring & Management

## Common Commands

```bash
# View logs
docker logs -f numinous_validator

# Check status
docker ps | grep numinous_validator

# Restart validator
docker compose -f docker-compose.validator.yaml --env-file .env.validator restart

# Update to new version
docker compose -f docker-compose.validator.yaml --env-file .env.validator pull
docker compose -f docker-compose.validator.yaml --env-file .env.validator up -d

# Stop validator
docker compose -f docker-compose.validator.yaml --env-file .env.validator down
```

---

# Troubleshooting

## Container Issues

```bash
# Check Docker is running
docker ps

# Check wallet files exist
ls -la ~/.bittensor/wallets/validator/

# Check logs for errors
docker logs numinous_validator 2>&1 | grep -i error

# Verify wallet structure
cat .env.validator | grep WALLET_NAME
ls -la ~/.bittensor/wallets/validator/coldkey
ls -la ~/.bittensor/wallets/validator/hotkeys/default
```