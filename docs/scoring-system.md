# Scoring System

Every validator scores every miner once a day, with a fixed formula and no model, so all validators compute the same scores from the same data. Your **credits** measure how much better your forecasts were than the house's own; your weight is computed from your credits.

---

# Credits

## How a statement is judged

After the deadline the backend opens every forecast. The judge then checks each statement against the news reported since the deadline: first on its next daily run, then regularly, and a last time at the horizon end, 30 days after the deadline. Each statement ends up:

- **correct** (`right` in `numi forecasts`), with the news event that made it correct and the day it was reported;
- **not correct** (`wrong`), with the judge's reason;
- or still **waiting** (`waiting`) until it is judged for good.

**Not correct at once:**
- the statement does not contain the subject the question asks about (and its counterparty, when the question has one): the topological condition;
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

## What you forecast: when it will appear

Each statement carries a `distribution` of **31 numbers**:

```text
p_1, p_2, ... p_30   the chance the statement first appears in the news on day 1, 2, ... 30 after the deadline
p_31                 the chance it does not appear within those 30 days
```

Day 1 is the first 24 hours after the round's deadline, day 2 the next 24 hours, and so on. A statement **appears** on the day the news event that makes it correct was reported; that day is its **outcome day** `y`. A statement with no qualifying news within the 30-day horizon has not appeared: `y = 31`.

The numbers are read as shares of their total, so they do not have to add up to exactly 1 (see [miner-setup.md](./miner-setup.md#3-write-the-forecast) for the format).

## The house

When a forecast opens, every statement in it gets the **house's** own 31 numbers `h_1 ... h_31`: how likely the statement was to appear on each day anyway, estimated from how often such developments were reported in the past. The house is fixed at that moment and does not depend on what you sent.

A statement that does not name the subject the question asks about gets no house numbers.

## The score of a statement

The underlying loss is the **Continuous Ranked Probability Score (CRPS)** evaluated on the daily grid, also called the **Ranked Probability Score (RPS)** for these ordered daily outcomes. It measures the squared error of the cumulative probabilities across days. We divide it by 30 and negate it to obtain the score `S`: lower CRPS is better, so higher `S` is better.

From a distribution, take the chance that the statement has appeared **by** day `d`:

```text
F_d(p) = p_1 + p_2 + ... + p_d          for d = 1 ... 30
```

The score compares that chance, day by day, with what happened: 0 before the outcome day, 1 from the outcome day on.

```text
S(p, y) = − (1 / 30) × Σ_(d = 1..30) ( F_d(p) − [y ≤ d] )²
```

`[y ≤ d]` is 1 when the statement had appeared by day `d` and 0 otherwise. `S` is 0 for a perfect forecast and −1 for the worst one. A forecast that puts its weight close to the day the statement appears scores close to 0; one that puts it far away, or on "not within 30 days", scores low.

Your **points** for the statement are your score minus the house's score on the same outcome:

```text
points(P) = S(p, y) − S(h, y)
```

This is the improvement in normalized CRPS over the house. The nonappearance cap and event-sharing rules below are additional reward adjustments; they are not part of CRPS itself.

- Better than the house: points above 0.
- The same numbers as the house: exactly 0.
- Worse than the house: points below 0.
- A statement with no house numbers: 0 points. It still counts as one of the bundle's statements.

## When a statement is scored

Credits are computed every day (see [When credits are computed](#when-credits-are-computed)), so a statement is scored before its 30 days are over:

- **A statement that has appeared** (on day 1 to 30) is scored on all 30 days at once, with the formula above. Every remaining day is already known, so its points no longer change.
- **A statement that has not appeared** is scored one day at a time, on the `D` full days (24 hours each, at most 30) that have passed since the deadline, as "not appeared yet" on each of them:

```text
points so far(P) = − (1 / 30) × Σ_(d = 1..D) ( F_d(p)² − F_d(h)² )
```

The division is always by 30, so these points grow towards the full score as the days pass. On the day the statement appears, it switches to the full 30-day score with its real outcome day: what it lost while waiting is replaced, not added.

## A statement that has not appeared can lose points but never earn them

While a statement has not appeared, its points so far are **capped at 0**:

```text
points(P) = min( 0, points so far(P) )       while y = 31
```

This also holds at the end: a statement that did not appear within its 30 days, or whose news was reported after day 30, ends with points at or below 0.

So saying "this will not appear within 30 days" never pays. If you think a development is less likely than the house does, you gain nothing by saying so: the scoring pays for calling what does happen, and when, better than the house. A statement the judge can never match earns nothing, however sure you are of it.

## A news event is shared among the miners paid for it

When several miners have points above 0 on the **same news event**, each one's points on it are divided by their number:

```text
points(P) = points(P) / m          when points(P) > 0
```

`m` is the number of miners **registered on the subnet at scoring time** that have a counted statement with points above 0 on that event, over every round still being scored. Only points above 0 are divided; points below 0 stay whole. A miner whose numbers only match the house's (points 0), or that is no longer registered, does not take a share.

A call that many miners make is worth less to each of them; a call only you make keeps its full points.

## De-duplication: a news event pays you once

If several of your statements are judged correct through the **same news event**, only one counts: the one in the bundle you sent first. The others are **voided**, whether they are in the same bundle, another question's bundle or another round's, in the current format or the previous one, and however they are worded. A voided statement earns nothing, costs nothing and is left out of `n_q`, so repeating a call is exactly like sending it once.

The event stays paid for as long as validators keep its data (90 days): a later repeat is still voided.

## Neither correct nor not correct

A statement judged `VOID` (see [Reasons](#reasons)) earns nothing, costs nothing and is left out of `n_q`, like a voided repeat.

## The formula

For each question `q` you answer, you send a **bundle** `Q_q` of statements (what `numi submit` calls a forecast). Each bundle earns:

```text
credits(Q_q) = Σ_(P ∈ Q_q) points(P) / √n_q
```

and your credits are the sum over your bundles, **floored at 0**:

```text
credits = max(0, Σ_q credits(Q_q))
```

| Symbol | Meaning |
|---|---|
| `points(P)` | The points of statement `P` at scoring time, after the cap and the sharing above |
| `n_q` | The number of statements in the bundle, still waiting ones included, minus the voided ones |
| which bundles count | **A bundle counts from its round's deadline until 60 days after it**: the 30 days of the forecast, then 30 more days with its final points |

## An example

A bundle of 3 statements, scored 5 full days after the deadline. The house gave each of them 0.1 for day 1, 0.1 for day 2 and 0.8 for "not within 30 days".

| Statement | Your numbers | What happened | Points |
|---|---|---|---|
| A | 0.2 on day 1, 0.6 on day 2, 0.2 on "not within 30 days" | appeared on day 2 | +0.579 |
| B | 0.3 on day 1, 0.7 on "not within 30 days" | not appeared yet | −0.009 |
| C | everything on "not within 30 days" | not appeared yet | 0 |

- **A** appeared, so it is scored on all 30 days. Your chance that it had appeared is 0.2 by day 1 and 0.8 from day 2 on: `S = −(0.2² + 29 × 0.2²) / 30 = −0.040`. The house had 0.1 by day 1 and 0.2 from day 2 on: `S = −(0.1² + 29 × 0.8²) / 30 = −0.619`. Points: `−0.040 − (−0.619) = +0.579`.
- **B** has not appeared, so only the 5 days passed count. You: `−(5 × 0.3²) / 30 = −0.015`. The house: `−(0.1² + 4 × 0.2²) / 30 = −0.006`. Points so far: `−0.015 − (−0.006) = −0.009`.
- **C** has not appeared either. You: 0. The house: −0.006. That would be +0.006, but a statement that has not appeared cannot earn: 0.

The bundle earns `(0.579 − 0.009 + 0) / √3` = **0.329**.

If one other registered miner also has points above 0 on the news event behind A, A's points are halved for both of you: `(0.579 / 2 − 0.009 + 0) / √3` = **0.162**.

What this means for you:
- **Send a statement only when you think it is more likely, or sooner, than the house does.** Matching the house earns 0; being less sure than the house about something that happens costs you.
- **Say when, not only whether.** Weight on the right day earns more than the same weight spread over the month.
- **A statement that does not appear costs you for the chance you gave it above the house's**, a little more each day, and can never earn.
- **A statement still waiting is one of the bundle's `n_q`** from the first day.

## Forecasts sent in the previous format

A forecast sent before this scoring was released, with one `probability` per statement, keeps the scoring it was sent under. Each of its judged statements has a baseline `b`, how likely it was to happen anyway, and:

```text
points(P) = 1 − b(P)   if P is judged correct
points(P) = − b(P)     if P is judged not correct
```

Each of those statements counts for 30 days from the day it was judged, and the bundle earns its points divided by `√n_q` as above. Nothing in the sections on the house, the cap or the sharing applies to them.

Your credits are the sum of the credits of all your bundles, in both formats, floored at 0. The previous format's statements leave the sum by themselves as their 30 days end. New forecasts are only accepted in the current format.

## Topological condition

Every statement must contain the entity the question asks about (and its counterparty, when the question has one). A statement that does not is judged not correct. This keeps a bundle about the question, not about the whole graph.

## What you can see, and when

- `numi forecasts` shows your own forecasts and how each statement was judged, as before.
- The **house's numbers** of a statement are public only once that statement is judged, on the leaderboard's forecast page and score page. While a statement is still waiting, its house numbers are not shown.

## What never counts

- An **invalid** bundle: it did not open, or did not follow the schema
- A bundle that **fails a validator check**: the sealed bytes do not match the hash received before the deadline, or the signature is not the miner's. Validators run this check when the whole forecast is released, once every statement in it is judged; a bundle that fails it stops counting from then on

## When credits are computed

Validators compute credits every day using a **15:00 UTC cutoff**. Waiting statements contribute their elapsed-day score, with positive points capped at 0 while they have not appeared. Judgments recorded before the cutoff determine which statements have resolved and their outcomes. A validator computes credits only after a results pull that started after 15:00, so all validators score the same data.

## Reasons

`numi forecasts` gives each judged statement its reason:

| Reason | Judged | Meaning |
|---|---|---|
| `REPORTED` | correct | News after the deadline reported this relationship |
| `MATCHED` | correct | A reported development is the act the statement states |
| `ASKED_COMPANY_NOT_NAMED` | not correct, at once | The statement does not name the question's subject on either side. It has no house numbers: 0 points, still counted |
| `ALREADY_HELD` | not correct, at once | This relationship was already known before the deadline |
| `NOT_NAMED` | not correct, at once | A party is not a named entity |
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
