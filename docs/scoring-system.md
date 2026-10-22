# Scoring System

Every validator scores every miner once a day, with a fixed formula and no model, so all validators compute the same scores from the same data. Your **credits** measure how much of what you forecast came true; your weight is computed from your credits.

---

# Credits

## How a statement is judged

After the deadline the backend opens every forecast. The judge then checks each statement against the news reported since the deadline: first on its next daily run, then regularly, and a last time at the horizon end. Each statement ends up:

- **correct** (`right` in `numi forecasts`), with the news event that made it correct and the day it was reported;
- **not correct** (`wrong`), with the judge's reason;
- or still **waiting** (`waiting`) until it is judged for good.

**Not correct at once:**
- the statement does not contain the company the question asks about (and its counterparty, when the question has one): the topological condition;
- a party is not named;
- the graph already held this relationship at the deadline;
- it repeats what the graph already held and states nothing new.

**Correct** when a news event reported after the deadline shows the development happened:
- With one of the relation types and both parties known to the graph, the check is direct: an event must report exactly that relationship between the two, in that direction unless the relation is symmetric, as more than talks or a rumour. Reported only as talks or a rumour, the statement keeps waiting.
- With free text, or a party the graph does not know, a model reads the news and decides whether what the statement says happened.
- When the article says what day the development happened, that day must be on or after the deadline's day. News that was public before the deadline does not count, even if the graph collected it later.

**Not correct at the horizon end:** a statement that is still waiting when the horizon ends.

`numi forecasts` shows how each of your statements was judged, and `numi graph` shows what the graph already held. A statement judged not correct, or never judged correct before the horizon ends, earns nothing. **Nothing ever subtracts.**

## The formula

For each question `q` you answer, you send a **bundle** `Q_q` of statements (what `numi submit` calls a forecast). Your credits are:

```text
credits = Σ_q  Σ_(p ∈ Q_q)  s(P_p) × 1(t − l_p < 30 days) × a(P_p)
```

| Symbol | Meaning |
|---|---|
| `s(P_p)` | 1 if the statement `P_p` is judged correct, 0 otherwise (in particular if it is never judged correct) |
| `l_p` | When `P_p` was judged correct (`judged_at`) |
| `t` | The moment of scoring: `1(t − l_p < 30 days)` is 1 while the correct statement is less than 30 days old, 0 after. **A correct statement counts for 30 days.** |
| `a(P_p)` | `1 / √#Q_q`, where `#Q_q` is the number of statements in the bundle |

In words, each bundle earns:

```text
(statements judged correct in the last 30 days) / √(statements in the bundle)
```

For example:

| Bundle | Credits |
|---|---|
| 1 statement, judged correct | 1 / √1 = **1.00** |
| 2 statements, 1 judged correct | 1 / √2 = **0.71** |
| 2 statements, both judged correct | 2 / √2 = **1.41** |

A second statement pays only when it is judged correct too, so add one only when you believe in it. A correct statement that leaves the 30-day window stops counting but stays one of the bundle's statements: a 2-statement bundle with one correct statement left in the window earns 0.71.

## Topological condition

Every statement must contain the entity the question asks about (and its counterparty, when the question has one). A statement that does not is judged not correct. This keeps a bundle about the question, not about the whole graph.

## De-duplication: a news event pays you once

If several of your statements are judged correct through the **same news event**, only one counts: the one in the bundle you sent first. The others are **voided**, whether they are in the same bundle, another question's bundle or another round's, and however they are worded. A voided statement is left out of both the correct statements and the statement count of its bundle, so repeating a call is exactly like sending it once.

The event stays paid after its statement leaves the 30-day window: a later repeat is still voided.

## What never counts

- An **invalid** bundle: it did not open, or did not follow the schema
- A bundle that **fails a validator check**: the sealed bytes do not match the hash received before the deadline, or the signature is not the miner's

## When credits are computed

Validators compute credits at **15:00 UTC** every day, counting only the statements judged correct before 15:00. A validator computes them only after a results pull that started after 15:00, so all validators score the same data.

---

# Weights

Validators set weights from the latest credits that are at least an hour old, so they all switch to a new day's credits together at 16:00 UTC.

Weights **extremise quadratically among the top 25**, plus a small share spread by rank:

```text
w_i = 0.98 × credits_i² / Σ_(j ∈ top 25) credits_j²   +   0.02 × (1 / r_i) / Σ_(r = 1..n) (1 / r)
```

- The first term only counts for the top 25 miners by credits.
- In the second term, `r_i` is miner `i`'s rank and `n` the number of miners with credits above 0.
- A miner with 0 credits gets no weight.
- **Ties** are broken by the newer registration, then by the lower uid.
- If **no miner** has credits, all weight goes to the subnet owner's uid.

The 2% keeps every miner with credits above every idle one: the chain deregisters the lowest-emission miners first, so without it a good miner just outside the top 25 would be deregistered before an idle one. And since both terms only decrease down the ranking, the 26th never gets more than the 25th.

**Example.** Two miners, with 3 and 1 credits:

| | First term (98%) | Second term (2%) | Weight |
|---|---|---|---|
| 3 credits (rank 1) | 9/10 × 98% = 88.2% | (1/1) / 1.5 × 2% = 1.33% | **89.5%** |
| 1 credit (rank 2) | 1/10 × 98% = 9.8% | (1/2) / 1.5 × 2% = 0.67% | **10.5%** |
