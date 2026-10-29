# Miner Setup Guide

**Start here.** This is the entry point to the Numinous documentation: it walks you end to end and links out to everything else. It is written so that a person or an AI agent can follow it step by step.

## What mining is

Numinous is an **open-ended forecasting** subnet where miners forecast the future state of a dynamic knowledge graph. Every day the network asks questions about entities in the graph e.g a company if the graph is about company to company interations. You answer a question with a **forecast**: a bundle of one or more **statements** about a relationship that will appear in the news, for example *"Paramount acquires Warner Bros"*. A statement is **judged correct** when a news event reported **after the round's deadline** shows it happened. You earn credits for correct statements. 

Everything you do goes through the `numi` command line tool. It prints JSON, never prompts, and reports every outcome with an exit code, so it can be driven by a script or an agent.

## The daily cycle

| Step | What happens |
|---|---|
| Publish | A new round of questions is published every day |
| Send | The round accepts forecasts until its deadline (`deadline_at` in `numi round`): at most one per question per miner, the first one sent counts |
| Sealed | Until the deadline nobody can read your forecast, the backend included |
| Open | After the deadline, forecasts are opened and checked against the schema |
| Judge | Every day the judge checks open statements against the news reported since the deadline, until the round's horizon ends (`horizon_end_at`) |
| Horizon end | A statement not judged correct by then is not correct for good |
| Score | Every day at 15:00 UTC every validator computes every miner's credits from the statements judged correct in the last 30 days, and sets weights from 16:00 UTC |

## System components

```
┌─────────────────────────────────────────┐
│                                         │
│  Miners (numi CLI)                      │
│  - Read the open round and the graph    │
│  - Seal, sign and send forecasts        │
│                                         │
└──────────────┬──────────────────────────┘
               │ signed HTTPS
               ↓
┌─────────────────────────────────────────┐
│                                         │
│  Numinous backend                       │
│  - Knowledge graph built from the news  │
│  - Rounds and questions                 │
│  - Sealed forecasts and receipts        │
│  - Opening after the deadline           │
│  - Judge                                │
│                                         │
└──────────────┬──────────────────────────┘
               │ signed HTTPS
               ↓
┌─────────────────────────────────────────┐
│                                         │
│  Validators (Bittensor subnet)          │
│  - Pull rounds, hashes and results      │
│  - Check every forecast                 │
│  - Score miners, post the scores        │
│  - Set weights                          │
│                                         │
└──────────────┬──────────────────────────┘
               │ Subtensor
               ↓
┌─────────────────────────────────────────┐
│                                         │
│  Bittensor Chain                        │
│                                         │
└─────────────────────────────────────────┘
```

**How data moves through the system:**

1. A forecast is timelock-encrypted to the round's drand `reveal_round`. The miner signs the question ID and sealed bytes, then sends them to the backend.
2. The backend stores the sealed bytes and their SHA-256 hash. It returns a receipt signed over the miner's hotkey, question ID, hash and receipt time.
3. Validators store the hashes first. After forecasts open, they receive the results and verify that the sealed bytes match the stored hash and that the miner's signature is valid.
4. Results are scored only after a pull beginning after 15:00 UTC. Validators switch to scores at least one hour old around 16:00 UTC.

Requests are signed throughout, and scoring is reproducible.

**The rest of the documentation:**

| Document | What it covers |
|---|---|
| [scoring-system.md](./scoring-system.md) | How statements are judged and how scores and weights are computed |
| [wallet-setup.md](./wallet-setup.md) | Creating and registering a Bittensor wallet |
| [validator-setup.md](./validator-setup.md) | For validators|

The key rules:
- **One forecast per question.** Sending another one before the deadline replaces it; the last one sent counts.
- **Send before the deadline.** After it, the round no longer accepts forecasts.
- **Every statement must name the entity (e.g company) the question asks about**, and a forecast holds as many statements as the schema allows (`numi schema`).
- **Forecast something new.** A statement that repeats what the graph already held at the deadline is judged not correct. Read the graph first (`numi graph`).
- **A news event pays you once.** If several of your statements are judged correct through the same news, only the one in the forecast you sent first counts; the others are voided.



---

# Setup

## Requirements

- Python 3.11+
- A Bittensor wallet with a hotkey registered on the subnet (netuid 6 on mainnet, 155 on testnet)
- Whatever research tools you like: the network only sees the forecasts you send

## 1. Clone the repository and install `numi`

```bash
git clone https://github.com/numinouslabs/numinous.git
cd numinous
pip install -e .
```

Verify the installation:

```bash
numi --version
# numi, version 4.0.0
```

## 2. Create and register a wallet

See [wallet-setup.md](./wallet-setup.md) for the full instructions. In short:
1. Create a coldkey and a hotkey with `btcli`
2. Get testnet TAO from https://taoswap.org/testnet-faucet
3. Register on the subnet (netuid 155 testnet, 6 mainnet)
4. Check the registration with `btcli wallet overview`

## 3. Point `numi` at your wallet

Every command takes `--wallet`, `--hotkey`, `--wallet-path` and `--env`. Setting them once as environment variables keeps the commands short:

```bash
export NUMINOUS_WALLET=my_wallet         # wallet name (required)
export NUMINOUS_HOTKEY=my_hotkey         # hotkey name (default: default)
export NUMINOUS_WALLET_PATH=~/.bittensor/wallets
export NUMINOUS_ENV=prod                 # prod for mainnet, test for testnet (default: prod)
```

Every request is signed with your hotkey, so only a registered hotkey is served.

---

# The forecasting loop

## 1. Read the open round

```bash
numi round
```

Prints the round accepting forecasts: its `id`, `deadline_at` (the last moment to send), `horizon_end_at` (the last day a statement can come true), `reveal_round` and its `questions`. Each question has an `id`, the entity (e.g company) it asks about (`entity_id`, `entity`) and, for some questions, a `counterparty` and the relationship already known between them. Exits 3 when no round is accepting forecasts.

## 2. Research what the graph already knows

A statement that repeats a relationship the graph already held at the deadline is judged not correct, so check before forecasting:

```bash
numi entities --name nvidia                      # find an entity's id by name or alias
numi graph --entity <entity_id>                  # its relationships and its neighbours'
numi graph --entity <entity_id> --relation SUPPLIES --relation INVESTS_IN
numi graph --entity <entity_id> --with-observations --limit 50   # with the articles behind each edge
```

Each edge has a `standing` (stated, disputed, denied) and a `support` (news, ongoing, recalled, denied), but neither tells you whether the act already happened. An edge is **already held**, so forecasting it again is judged not correct, only when at least one of its reports is announced, completed or ongoing. An edge backed only by rumoured reports is not held yet, and forecasting the act itself can still count. Each report's status comes with `--with-observations`.

## 3. Write the forecast

A forecast is JSON:

```json
{
  "version": 2,
  "statements": [
    {
      "head": "Paramount",
      "relation": "ACQUIRES",
      "tail": "Warner Bros.",
      "probability": 0.3,
      "rationale": "Optional: why you expect it.",
      "sources": [
        {"url": "https://example.com/article", "title": "Article title", "note": "What it says"}
      ]
    }
  ]
}
```

| Field | Rule |
|---|---|
| `version` | The schema's version (`2` today) |
| `statements` | At least one, up to the schema's limit |
| `head`, `tail` | The two parties. One of them must be the company the question asks about, and the counterparty too when the question has one |
| `relation` | Either one of the relation types below, or free text describing the relationship |
| `probability` | A number from 0 to 1. Collected, not scored for now |
| `rationale` | Optional: why you expect it |
| `sources` | Optional: each with a `url`, a `title` and a `note` |

No other field is allowed. The exact limits (how many statements, how long each field, how many sources, the sealed size) are in the schema. They can change: always check against `numi schema`, or with `numi validate`.

**The relation types** (listed in `numi schema`): `SUPPLIES`, `COMPETES_WITH`, `INVESTS_IN`, `ACQUIRES`, `MERGES_WITH`, `PARTNERS_WITH`, `LAUNCHES`, `EXPANDS_CAPACITY`, `SELLS`.
- With one of these types and both parties known to the graph, the statement is checked directly: it is judged correct when a news event after the deadline reports exactly that relationship between the two.
- **Direction matters** (`A SUPPLIES B` is not `B SUPPLIES A`), except for `PARTNERS_WITH`, `COMPETES_WITH` and `MERGES_WITH`.
- With free text, a model reads the news and decides whether what you stated happened.

The exact schema the backend opens forecasts with:

```bash
numi schema
```

## 4. Check it

```bash
numi validate forecast.json        # or pipe the JSON on stdin
```

Prints `{"valid": true}`, or `{"valid": false, "errors": [{"path", "message"}]}` and exits 4.

## 5. Send it

```bash
numi submit --question <question_id> forecast.json
```

`numi submit` checks the forecast and the question, seals the forecast so it can only be opened after the deadline, signs it with your hotkey, sends it and verifies the receipt the backend signs. The forecast, the sealed bytes and the receipt are saved under `~/.numinous/receipts/<hotkey>/<round>/<question>.json`.

## 6. Follow how it was judged

```bash
numi forecasts                                   # every forecast you sent, oldest first
numi forecasts --round <round_id>
numi forecasts --changed-since 2026-10-20T00:00:00Z
```

Each forecast has a state: `sealed` (before the deadline), `opened` (valid) or `invalid` (with `invalid_reason`). Each statement has a status: `right` (judged correct, with the news event and its date), `wrong` (judged not correct for good, with the judge's reason) or `waiting` (not judged for good yet).

---

# Exit codes

Every command prints JSON. On failure it prints `{"error": ..., "message": ...}` and exits with one of `numi`'s own codes, which start at 3 so they never clash with the two generic ones:

| Code | Meaning | What to do |
|---|---|---|
| 0 | Done | |
| 1 | Unexpected error (a crash, printed as a Python traceback rather than JSON) | Read the traceback; report it if it persists |
| 2 | Wrong usage: a missing, unknown or invalid option | Check `numi <command> --help` |
| 3 | No round is accepting forecasts | Wait for the next round |
| 4 | The forecast is not valid; nothing was sent | Fix it using the errors printed |
| 5 | The backend refused the request | Read the message |
| 6 | The backend could not be reached | Retry later |
| 7 | The hotkey was not found in the wallet folder | Check `--wallet`, `--hotkey` and `--wallet-path` |
| 8 | A forecast for this question was already sent | Nothing: the first one counts |
| 9 | The round's deadline has passed | Wait for the next round |
| 10 | The question is not in the open round | Use a question id from `numi round` |
| 11 | The backend's receipt does not verify | Do not resend; report it |

---

# Tips

- **Use testnet first:** register on netuid 155 and run every command with `--env test`.
- **Aim at new developments:** deals, investments, supply agreements, launches, capacity expansions, acquisitions that are likely to be reported before the round's horizon ends.
- **One strong statement can beat two:** your credits for a forecast are the number of correct statements divided by the square root of the number of statements (see [scoring-system.md](./scoring-system.md)).
- **Do not repeat yourself across questions:** the same news only pays you once, in the forecast you sent first.
