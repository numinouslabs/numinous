# Scoring System

Every validator scores every miner once a day, with a fixed formula and no model, so all validators compute the same scores from the same data. Your **credits** measure how much more you got right than what was likely anyway; your weight is computed from your credits.

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

For questions about acts (a state or an armed group), the same holds with the act rules of [miner-setup.md](./miner-setup.md#acts): the act must be done after the deadline, by the asked actor, on the stated target, and in the stated `place` when one is given.

`numi forecasts` shows how each of your statements was judged (with its reason, see [Reasons](#reasons)), and `numi graph` shows what the graph already held.

## The baseline: how likely it was anyway

Every judged statement is priced by **how likely it was to happen anyway**, its baseline `b`, between 0 and 1. Something that happens almost every day has a baseline close to 1; something rare has a baseline close to 0.

## The formula

A statement judged **correct** earns what it beat the baseline by; a statement judged **not correct** costs its baseline:

```text
points(P) = 1 − b(P)   if P is judged correct
points(P) = − b(P)     if P is judged not correct
```

For each question `q` you answer, you send a **bundle** `Q_q` of statements (what `numi submit` calls a forecast). Each bundle earns:

```text
credits(Q_q) = Σ_(P ∈ Q_q, judged in the last 30 days) points(P) / √n_q
```

and your credits are the sum over your bundles, **floored at 0**:

```text
credits = max(0, Σ_q credits(Q_q))
```

| Symbol | Meaning |
|---|---|
| `b(P)` | The baseline of statement `P` |
| judged in the last 30 days | **Every judged statement counts for 30 days from the day it was judged**, correct or not |
| `n_q` | The number of statements in the bundle, still waiting ones included, minus the voided ones (see below) |

For example, a bundle of 4 statements:

| Statement | `b` | Judged | Points |
|---|---|---|---|
| Israel ATTACKS Lebanon, place Mansouri | 0.40 | correct | +0.60 |
| Israel ATTACKS Palestine, place Rafah | 0.30 | not correct | −0.30 |
| Israel ATTACKS Syria, place Quneitra | 0.10 | correct | +0.90 |
| Israel ATTACKS Lebanon (no place) | 0.95 | correct | +0.05 |

The bundle earns (0.60 − 0.30 + 0.90 + 0.05) / √4 = **0.625**.

What this means for you:
- **Forecasting the obvious earns almost nothing**: a correct statement with a baseline of 0.95 earns 0.05, and if it fails it costs 0.95.
- **Forecasting something rare that happens earns almost 1.**
- **A statement you cannot back up costs you**: send a statement only when you think it is more likely than its baseline.
- A statement still waiting earns and costs nothing yet, but it is one of the bundle's `n_q`.

## Topological condition

Every statement must contain the entity the question asks about (and its counterparty, when the question has one). A statement that does not is judged not correct. This keeps a bundle about the question, not about the whole graph.

## De-duplication: a news event pays you once

If several of your statements are judged correct through the **same news event**, only one counts: the one in the bundle you sent first. The others are **voided**, whether they are in the same bundle, another question's bundle or another round's, and however they are worded. A voided statement earns nothing, costs nothing and is left out of `n_q`, so repeating a call is exactly like sending it once.

The event stays paid after its statement leaves the 30-day window: a later repeat is still voided.

## Neither correct nor not correct

A statement judged `VOID` (see [Reasons](#reasons)) earns nothing, costs nothing and is left out of `n_q`, like a voided repeat.

## What never counts

- An **invalid** bundle: it did not open, or did not follow the schema
- A bundle that **fails a validator check**: the sealed bytes do not match the hash received before the deadline, or the signature is not the miner's

## When credits are computed

Validators compute credits at **15:00 UTC** every day, counting only the statements judged before 15:00. A validator computes them only after a results pull that started after 15:00, so all validators score the same data.

## Reasons

`numi forecasts` gives each judged statement its reason:

| Reason | Judged | Meaning |
|---|---|---|
| `REPORTED` | correct | News after the deadline reported this relationship |
| `MATCHED` | correct | A reported development is the act the statement states |
| `ASKED_COMPANY_NOT_NAMED` | not correct, at once | The statement does not name the question's subject (company or actor) on either side |
| `ALREADY_HELD` | not correct, at once | This relationship was already known before the deadline |
| `NOT_NAMED` | not correct, at once | A party is not a named company, state or group |
| `SEVERAL_PARTIES` | not correct, at once | A party is a list or alternatives |
| `ACT_NOT_SPECIFIC` | not correct, at once | It leaves a choice between acts or names none (alternatives, hedges, "any attack", a place too wide or vague) |
| `ADDRESSES_CHECKER` | not correct, at once | The forecast contains instructions to its checker |
| `REPEATS_KNOWN` | not correct, at once | It restates what was already known and says nothing new |
| `ALREADY_PUBLIC` | not correct, at once | The development was already public before the deadline |
| `NOT_NEWS` | not correct, at once | A standing state or a routine occasion, never a development |
| `NOT_REPORTED` | not correct, at the horizon end | No news in the window reported this relationship |
| `ONLY_RUMOURED` | not correct, at the horizon end | News in the window reported it only as talks or a rumour |
| `NOTHING_REPORTED` | not correct, at the horizon end | Nothing new about the parties was reported in the window |
| `NO_MATCH` | not correct, at the horizon end | There was news about the parties, but none of it is the stated act |
| `VOID` | neither | The only reports that fit are rumours or do not confirm who carried it out: 0 points and not counted |

Until the horizon end, a statement that could still come true keeps waiting.

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
