<div align="center">

# **Numinous**



[Discord](https://discord.gg/qKPeYPc3) • [Dashboard](https://app.hex.tech/1644b22a-abe5-4113-9d5f-3ad05e4a8de7/app/Numinous-031erYRYSssIrH3W3KcyHg/latest) • [Website](https://numinouslabs.io/) • [Twitter](https://x.com/numinous_ai) •
[Network](https://taostats.io/subnets/6/chart)
---

</div>

### 👉 New here? Start with the **[Miner Setup Guide](docs/miner-setup.md)**, the entry point to all the documentation.

## Introduction

Numinous (Subnet 6) is a **graph forecasting** subnet. It forecasts how the network of companies and their relationships will change: who will invest in, supply, partner with, acquire or compete with whom.

The Numinous backend builds a **knowledge graph** of companies and their relationships from the news. Every day the subnet asks questions about companies, and miners answer each question with a bundle of statements about relationships that will appear next. A statement is judged correct when the news shows it happened after the question's deadline.

The key principles of the subnet are:

  * **Verifiable:** every forecast is sealed before the deadline, judged against public news, and scored with a fixed formula that any validator can recompute.
  * **Forward-looking:** only new developments count. A statement that repeats what the graph already knew earns nothing.

-----

## 🏗 System Architecture

Every day follows the same cycle: **Questions → Sealed forecasts → Opening → Judging → Scoring → Weight setting.**

### Key Components

  * **The knowledge graph:** companies and their relationships, built from the news by the Numinous backend.
  * **`numi`:** the miners' command line tool to read the open round and the graph, and to seal, sign and send forecasts.
  * **The judge:** checks every statement against the news reported after the deadline.
  * **Validators:** check every forecast against its hash and signature, score every miner, and set weights.

📖 **[Read how the system works](docs/miner-setup.md#system-components)**

-----

## ⚠️ Rules & Scoring

### Rules

1.  **Rounds:** a new round of questions is published every day and accepts forecasts until its deadline.
2.  **One forecast per question:** the first one sent counts. Each of its statements names the company the question asks about.
3.  **Sealed until the deadline:** nobody can read a forecast before the deadline, the backend included.
4.  **New developments only:** a statement comes true through news reported after the deadline, until the round's horizon ends.

### Scoring

Each bundle earns `correct statements / √statements` in **credits**, counting statements judged correct in the last 30 days. A news event pays a miner once: repeats are voided. Validators compute credits daily at 15:00 UTC, and weights **extremise quadratically among the top 25** (98% by credits squared) with **2%** spread by rank over every miner with credits.

⚠️ **[Read the miner guide and rules](docs/miner-setup.md)** • 📊 **[Read the scoring system](docs/scoring-system.md)**


-----

## 🚀 Getting Started

### For Miners

Forecast the knowledge graph and compete for the daily reward pool.

  * [**Miner Setup Guide**](docs/miner-setup.md): **start here.** Installation, wallet registration and the `numi` forecasting loop.
  * [**Scoring System**](docs/scoring-system.md): how you are scored and how weights are set.

### For Validators

Check every forecast, score every miner and set weights. No special hardware needed.

  * [**Validator Setup Guide**](docs/validator-setup.md): requirements and node configuration.

-----

## 📄 License

This repository is licensed under the MIT License.
