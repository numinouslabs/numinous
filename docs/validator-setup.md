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
- **Testnet (netuid 155):** Minimal TAO (available from faucet)
- **Mainnet (netuid 6):** 10,000 TAO minimum stake

---

# Architecture Overview

**How Validators Work:**

1. **Pull Events:** Fetch binary prediction events from Numinous API
2. **Pull Agent Code:** Download Python agent code submitted by miners via API
3. **Execute in Sandboxes:** Run each miner's agent in isolated Docker containers
4. **Collect Predictions:** Store predictions with run_id (execution UUID) and version_id (agent version)
5. **Score Performance:** Compute a Brier score over miner predictions
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
cd /root
git clone https://github.com/numinouslabs/numinous.git
cd numinous
```

## 3. Setup Wallet

Create and register your validator wallet. See [wallet-setup.md](./wallet-setup.md) for detailed instructions.

```bash
# Create wallet
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default

# Copy wallet to repository
mkdir -p /root/numinous/.bittensor/wallets
cp -r ~/.bittensor/wallets/validator /root/numinous/.bittensor/wallets/

# Register on subnet
# Testnet: --netuid 155 --subtensor.network test
# Mainnet: --netuid 6 --subtensor.network finney
btcli subnet register --wallet.name validator --wallet.hotkey default --netuid 6 --subtensor.network finney
```

## 4. Configure Environment Variables

Create environment file from template:

```bash
cd /root/numinous
cp .env.prod.example .env.prod
```

Edit the file with your configuration:

```bash
vim .env.prod
```

**Required variables:**

Contact the team on Discord for IAM user credentials to access the ECR registry.

```bash
# AWS ECR Configuration
# Contact the SN6 team for registry URL
AWS_ECR_REGISTRY=<ecr-registry-url>

# Wallet Configuration
# Must match directory names under .bittensor/wallets/
WALLET_NAME=validator
WALLET_HOTKEY=default
```

**Save and exit**

## 5. Authenticate with ECR

Contact the team on Discord for IAM user credentials to access the ECR registry.

```bash
# Install AWS CLI
sudo apt-get install -y awscli

# Configure with provided IAM credentials
aws configure
# AWS Access Key ID: <provided-by-team>
# AWS Secret Access Key: <provided-by-team>
# Default region: <provided-by-team>
# Default output format: json

# Login to ECR
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin ${AWS_ECR_REGISTRY}
```

## 6. Start Validator

```bash
cd /root/numinous

# Pull and start validator
docker compose -f docker-compose.prod.yaml --env-file .env.prod pull
docker compose -f docker-compose.prod.yaml --env-file .env.prod up -d

# View logs
docker logs -f if_validator_prod
```

# Monitoring & Management

## Common Commands

```bash
# View logs
docker logs -f if_validator_prod

# Check status
docker ps | grep if_validator_prod

# Restart validator
docker compose -f docker-compose.prod.yaml --env-file .env.prod restart

# Update to new version
docker compose -f docker-compose.prod.yaml --env-file .env.prod pull
docker compose -f docker-compose.prod.yaml --env-file .env.prod up -d
```

---

# Troubleshooting

## Container Issues

```bash
# Check Docker is running
docker ps

# Check wallet files exist
ls -la /root/numinous/.bittensor/wallets/validator/

# Check logs for errors
docker logs if_validator_prod 2>&1 | grep -i error

# Verify wallet structure
cat .env.prod | grep WALLET_NAME
ls -la /root/numinous/.bittensor/wallets/validator/coldkey
ls -la /root/numinous/.bittensor/wallets/validator/hotkeys/default
```

## ECR Authentication Issues

```bash
# Re-authenticate
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin ${AWS_ECR_REGISTRY}

# Verify AWS credentials
aws sts get-caller-identity
```