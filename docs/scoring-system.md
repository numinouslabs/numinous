# Scoring System

This document explains the scoring and payout system shown on the [Numinous leaderboard](https://leaderboard.numinouslabs.io/miner-console/setup?step=-1&coldkey=&hotkey=&browse=false).

## Only one active pool and track

**At launch there is exactly one pool: the re-forecasting pool on the SIGNAL track.** The entire daily emission is distributed against it.

Everything else described further down this document — the global pool, the `(topic, metric)` domain pools, PNL, the geopolitical and finance pools, the Reasoning pool, and Hawkes source scoring — is **not currently scored and earns nothing.** The code still accepts a MAIN-track agent submission, and the endpoints still exist, but no emissions are allocated to them.

The Reasoning pool is expected to be re-introduced in the near future. Agents may already return `reasoning` alongside their forecast, and it is stored, so returning it now costs nothing and prepares you for that.

If you are here to mine, the re-forecasting pool is the only thing that matters.

## Re-Forecasting Pool

Every live event is re-forecast by every agent, every interval. Each forecast is scored against the market's own price at the moment it was made:

```text
S = (prediction - target)^2 - (market_price - target)^2
```

where `target` is the market price 7 days later, or the realized outcome if the market resolved before then.

- **Lower is better; negative means you beat the market.**
- Matching the market scores exactly `0` — it is the baseline, not the goal.
- A forecast becomes scoreable 7 days after it is made, once the price it is measured against exists. Scores are then averaged over a rolling 7-day window.
- Your average is taken over **events**, one score per event, and its denominator is the set of events **you** actually forecast — skipping events does not dilute it. Absence is punished by coverage instead.
- A miner must cover at least **85%** of its available `(event, interval)` cells over a rolling 14-day window, or it earns zero regardless of score.
- A new miner must also accumulate three scored days before it can rank, so first earnings land around `T+10`.

### Confidence penalty for new miners

A miner is ranked and paid on a pessimistic bound of its mean, not the raw mean. The bound adds a penalty that depends only on how many scored forecast days the miner has accumulated, and it decays to zero within about two weeks:

```text
ucb    = mean + pen(T)
pen(T) = z * lambda * sqrt(corr_sum[T] / T) * exp(-max(0, T - H0) / tau)
```

- `T` is the number of scored forecast days for this `(uid, hotkey)`. It resets on re-registration.
- `z = 1.6449`, the one-sided 95% normal quantile.
- `lambda` is the field's day-to-day volatility of the mean score, re-estimated every scoring run (see below).
- `corr_sum[T]` accounts for consecutive forecast days sharing most of their 7-day scoring window, so `T` days are worth fewer than `T` independent samples. It is derived from the lag correlation `rho(l) = 0.95 * max(0, 7 - l) / 7`, which is fitted once field-wide and decays linearly to zero at the horizon:

```text
corr_sum[T] = 1 + 2 * sum_{l=1}^{T-1} (1 - l/T) * rho(l)
```

- `H0 = 3` days of full protection: through the first three scored days the penalty is essentially the full day-one gate.
- `tau = 2.5` days sets the release: after `H0` the penalty decays exponentially, so a newcomer with real skill passes a weaker incumbent within about seven scored days.

The penalty is a fixed handicap schedule: no per-miner statistic enters, so two miners with the same tenure carry the same penalty regardless of how their own scores look. Because lower is better, the penalty holds a debut miner below the ranks its raw mean would earn until its standing rests on enough days.

**How `lambda` is estimated.** Each scoring run builds, per `(uid, hotkey)`, the series of daily mean point scores, keeping only days with at least 50 scored events. The variance of that series is bias-corrected for the same day-to-day correlation, since the sample mean of a correlated series absorbs `corr_sum` days of noise rather than one:

```text
g = var(m_1..m_T, ddof=1) * (T - 1) / max(T - corr_sum[T], 1)
```

Every miner with `T >= 4` scored days is included, and `lambda` is the 95th percentile of `sqrt(g)` across them. The tail, not the median, is what prices an unknown newcomer's risk; including short-history miners keeps the newcomers who matter inside that tail.

Illustrative schedule with `lambda = 0.024`, so the day-one gate `z * lambda` is about `0.039`:

| Scored days `T` | 1 | 3 | 5 | 7 | 10 | 14 |
|---|---|---|---|---|---|---|
| `pen(T)` | 0.039 | 0.036 | 0.015 | 0.006 | 0.002 | < 0.001 |

### Splitting the pool

Unlike the inactive pools below, the re-forecasting pool is **not winner-takes-all**. Among miners who clear coverage, each miner's take is the squared distance between its bound (`ucb` above) and the 21st-best bound, and weights are proportional to those takes:

```text
take   = max(baseline - ucb, 0)^2        baseline = 21st-best ucb
weight = take / sum(takes)
```

So roughly the top 20 earn, and what pays is the size of your margin rather than your rank. A decisive leader is rewarded for its margin; a field separated by rounding error splits close to evenly. If fewer than 21 miners qualify the baseline becomes the worst qualifying score, which means the last-placed miner always takes exactly zero — worth knowing during ramp-up. If nobody clears the gate at all, the whole emission burns.

For the full rules, the selection criteria, and the memory contract, see [subnet-rules.md](./subnet-rules.md). For the system-level mechanics, see [architecture.md](./architecture.md#scoring). For the reasoning behind the design, see [The Belief Curve That Front-Runs the Price](https://numinouslabs.io/blog).

---

# Inactive Pools

> **Not currently scored.** Everything below this line describes the previous multi-pool payout system. It is retained for reference and does not currently allocate any emissions.

## One Global Contest Plus K Domain Contests

The network can be viewed as one global contest plus `K` smaller contests defined by `(topic, metric)` pairs.

Each day, the subnet emits a total reward pool `E`.

- Global pool: `alpha_0 * E`
- `(topic, metric)` pools: `alpha_(t,r) * E` for each topic `t` and metric `r`

These pool weights satisfy:

```text
alpha_0 + sum_(t,r) alpha_(t,r) = 1
```

This means the full daily emission is split across:

- one global leaderboard
- multiple domain-specific leaderboards

## Winner-Takes-All Per Pool

Each pool is winner-takes-all.

For a given pool, the miner with the best ranking receives the full allocation for that pool. Using the ranking notation `r`, the final payout for miner `m` is:

```text
R(m) =
  alpha_0 * E * 1[m = argmin r_(m,global)]
  + sum_(t,r)^K alpha_(t,r) * E * 1[m = argmin r_(m,t,r)]
```

In plain terms:

- the best miner on the global leaderboard wins the global pool
- the best miner inside each `(topic, metric)` leaderboard wins that pool
- a miner can win multiple pools in the same day

## Metrics

The main metrics currently used for ranking are:

- `PNL`
- `Brier Score`
- `Reasoning` — a separate quality+calibration score on miner-submitted reasoning. When this pool was active it took 25% of emissions on the Information track and 20% on the Signals track. The `Reasoning` column on the leaderboard is a 70/30 weighted average of a rubric quality score and an implied-probability Brier score. See [reasoning-scoring.md](./reasoning-scoring.md) for the full rubric, the extractor prompt, and the 2,500-character reasoning length cap.

## PNL Metric

The `PNL` metric is used in the `Sport x PNL` pool.

The intuition is simple: we simulate a `$1` trade based on the miner's forecast relative to the Polymarket price at the time the forecast is made.

Let:

- `p_t` be the Polymarket price at forecast time
- `p_i` be the miner forecast

### Case 1: Market Price Above Miner Forecast

If `p_t > p_i`, the market is pricing the event above the miner forecast, so the simulated position is on the `NO` side.

- If the event does **not** happen, the score is:

```text
1 / (1 - p_i) - 1
```

- If the event **does** happen, the score is:

```text
-1
```

### Case 2: Market Price Below Miner Forecast

If `p_t < p_i`, the market is pricing the event below the miner forecast, so the simulated position is on the `YES` side.

- If the event **does** happen, the score is:

```text
1 / p_i - 1
```

- If the event does **not** happen, the score is:

```text
-1
```

### Interpretation

This metric rewards miners when their forecast identifies a profitable directional disagreement with the market.

- a correct position generates positive `PNL`
- an incorrect position loses the full simulated `$1`

## Geopolitical Pool

In the geopolitical pool, some events are resolved against the Polymarket price rather than against the final binary outcome.

These events are evaluated at a time `T`. If the Polymarket price at that time is `p_T`, then the miner score is:

```text
(p_i - p_T)^2
```

This is a squared error metric:

- lower is better
- miners are rewarded for matching the market-implied probability at the evaluation time `T` which is assumed to be the true probability

## Reference LLM

The finance pool includes long-term events where a reference agent is used instead of directly resolving against the market price (to prevent manipulation in thin markets).

At time `T`:

- the reference agent receives `p_T` as input
- the reference agent also performs its own web search
- the reference agent outputs an updated forecast `p_T'`

The miner score is then:

```text
(p_i - p_T')^2
```

This lets the system score long-duration finance events against a standardized reference forecast that combines:

- the market state at time `T`
- fresh external information gathered by the reference agent

## Hawkes Source Scoring

> The Information track this describes is **not active** and allocates no emissions.

The goal of the Hawkes scoring on the **Information track**  is to incentivise miners to surface high impact and relevant sources for a given future event. The idea is to use the maths of the Hawkes processes to translate a set of sources into a probability.

### 1. What you submit

For each forecast, your agent submits up to **20 sources** per event. Each source has:

| Field | Notes |
|---|---|
| `url` | Must be reachable. URLs that 4xx/5xx or never resolve are dropped |
| `source_timestamp` | When the source event happened in the world. **Must not be older than 36 hours.** Anything older is dropped before scoring. Future timestamps are also dropped. |
| `impact_bucket` | One of `LOW`, `MEDIUM`, `HIGH`, `EXTREME` — see coefficient table below. |
| `persistence_bucket` | One of `FLASH`, `SHORT`, `MEDIUM`, `LONG` — see coefficient table below. |
| `direction` | `UP` / `DOWN` / `NEUTRAL` (informational). |
| `reasoning` | ≤ 2500 chars, why this source is informative. |

#### Submission format

Each source your agent emits must conform to this schema (the validator will batch your sources into a `POST /v1/sources` request to the backend):

```json
{
  "url": "https://www.reuters.com/world/...",
  "source_type": "news_article",
  "direction": "up",
  "source_timestamp": "2026-04-29T08:42:00+00:00",
  "impact_bucket": "high",
  "persistence_bucket": "medium",
  "reasoning": "Why this source moves the YES probability for this event. Reference the specific claim, the publication time, and how the magnitude of the move maps to your impact bucket. ≤ 2500 chars."
}
```

**Field constraints**

| Field | Type | Constraint |
|---|---|---|
| `url` | string | required, ≤ 2048 chars, must be publicly reachable |
| `source_type` | string | required, free‑form (e.g. `"news_article"`, `"press_release"`, `"social_post"`, `"market_data"`, `"government_filing"`) |
| `direction` | enum | required, one of `"up"` / `"down"` / `"neutral"` |
| `source_timestamp` | ISO‑8601 datetime | required, timezone‑aware UTC, **must be ≤ 36 h old and not in the future** |
| `impact_bucket` | enum | required, one of `"low"` / `"medium"` / `"high"` / `"extreme"` |
| `persistence_bucket` | enum | required, one of `"flash"` / `"short"` / `"medium"` / `"long"` |
| `reasoning` | string | required, ≤ 2500 chars |

**Per‑event submission**

Your agent's full per‑event output is a list of these source objects. The list:

- must contain **effectively at least 3** to be scored — see §4,
- is capped at **20 sources** per event,
- has no required ordering — the scorer sorts by `source_timestamp` internally.

Sketch of what a complete agent return for one event looks like:

```json
{
  "event_id": "1f8a…",
  "prediction": 0.74,
  "reasoning": "Top-level rationale tying the sources together.",
  "sources": [
    { "url": "...", "source_type": "news_article", "direction": "up",   "source_timestamp": "...", "impact_bucket": "high",    "persistence_bucket": "medium", "reasoning": "..." },
    { "url": "...", "source_type": "press_release", "direction": "up",   "source_timestamp": "...", "impact_bucket": "extreme", "persistence_bucket": "short",  "reasoning": "..." },
    { "url": "...", "source_type": "social_post",   "direction": "down", "source_timestamp": "...", "impact_bucket": "low",     "persistence_bucket": "flash",  "reasoning": "..." }
  ]
}
```

**Validation rules enforced on the backend**

- A source which is outside the 36 h window is removed
- A source with an unreachable URL is removed
- A source with `relevance < 0.2` from the LLM judge is removed
- Re‑submitting the same source overwrites the previous row
- A miner needs at least 3 sources passing the above filters


### 2. From your buckets to a Hawkes model

The two bucket labels map deterministically to coefficients of a self‑exciting (Hawkes) point process according to a conversion table which is kept hidden in order to prevent overfitting:

**Impact bucket → α (excitation amplitude)**

**Persistence bucket → β (decay time constant, in hours)**


So when you choose `(impact, persistence)`, you are literally writing one term of a Hawkes kernel:

```
φ_i(t) = α_i · exp(-(t - t_i) / β_i)     for t ≥ t_i
```

Higher impact = bigger jump in implied probability when the source fires. Higher persistence = the bump decays over a longer horizon.

### 3. Relevance check (LLM gate)

Each `(scraped url, reasoning)` is scored 0–1 by an LLM judge for how relevant the source is to the event. Your **effective alpha** is:

```
α_effective_i = relevance_i · α(impact_bucket_i)
```

Sources with `relevance < 0.2` are dropped. 

### 4. The Hawkes curve

Sort surviving sources by `source_timestamp`. For each prefix of `k = 3, 4, …, n` sources the system evaluates the model at `t = source_timestamp` of source #k:

```
Compensator(t)  = μ · t  +  Σ_{i=1..k} α_effective_i · β_i · (1 − exp(-(t − t_i)/β_i))
Probability(t)  = 1 − exp(-Compensator(t))
```

`μ = 0.001` is the baseline intensity. `Probability(t)` is your model's implied YES probability *after seeing the first k sources*. Each `k` produces one curve point.

### 5. The curve point is scored against Polymarket

For each curve point at time `t`, the system looks up the Polymarket YES price on the event's `condition_id` at `t` and at `t + 72 h` (Δ = 72 hours), and computes a **value‑added Brier**:

```
score_per_point = ( price(t)         − price(t + 72h) )²    
                − ( probability(t)   − price(t + 72h) )²    
```

Miners have to beat the trivial "Polymarket price stays where it is" baseline at the 72h horizon. The curve points are averaged into a per‑submission score.
