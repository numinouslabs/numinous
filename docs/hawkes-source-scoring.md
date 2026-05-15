# Hawkes Source Scoring

The goal of the Hawkes scoring on the **Information track**  is to incentivise miners to surface high impact and relevant sources for a given future event. The idea is to use the maths of the Hawkes processes to translate a set of sources into a probability.

## 1. What you submit

For each forecast, your agent submits up to **20 sources** per event. Each source has:

| Field | Notes |
|---|---|
| `url` | Must be reachable. URLs that 4xx/5xx or never resolve are dropped |
| `source_timestamp` | When the source event happened in the world. **Must not be older than 36 hours.** Anything older is dropped before scoring. Future timestamps are also dropped. |
| `impact_bucket` | One of `LOW`, `MEDIUM`, `HIGH`, `EXTREME` — see coefficient table below. |
| `persistence_bucket` | One of `FLASH`, `SHORT`, `MEDIUM`, `LONG` — see coefficient table below. |
| `direction` | `UP` / `DOWN` / `NEUTRAL` (informational). |
| `reasoning` | ≤ 2500 chars, why this source is informative. |

### Submission format

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


## 2. From your buckets to a Hawkes model

The two bucket labels map deterministically to coefficients of a self‑exciting (Hawkes) point process according to a conversion table which is kept hidden in order to prevent overfitting:

**Impact bucket → α (excitation amplitude)**

**Persistence bucket → β (decay time constant, in hours)**


So when you choose `(impact, persistence)`, you are literally writing one term of a Hawkes kernel:

```
φ_i(t) = α_i · exp(-(t - t_i) / β_i)     for t ≥ t_i
```

Higher impact = bigger jump in implied probability when the source fires. Higher persistence = the bump decays over a longer horizon.

## 3. Relevance check (LLM gate)

Each `(scraped url, reasoning)` is scored 0–1 by an LLM judge for how relevant the source is to the event. Your **effective alpha** is:

```
α_effective_i = relevance_i · α(impact_bucket_i)
```

Sources with `relevance < 0.2` are dropped. 

## 4. The Hawkes curve

Sort surviving sources by `source_timestamp`. For each prefix of `k = 3, 4, …, n` sources the system evaluates the model at `t = source_timestamp` of source #k:

```
Compensator(t)  = μ · t  +  Σ_{i=1..k} α_effective_i · β_i · (1 − exp(-(t − t_i)/β_i))
Probability(t)  = 1 − exp(-Compensator(t))
```

`μ = 0.001` is the baseline intensity. `Probability(t)` is your model's implied YES probability *after seeing the first k sources*. Each `k` produces one curve point.

## 5. The curve point is scored against Polymarket

For each curve point at time `t`, the system looks up the Polymarket YES price on the event's `condition_id` at `t` and at `t + 72 h` (Δ = 72 hours), and computes a **value‑added Brier**:

```
score_per_point = ( price(t)         − price(t + 72h) )²    
                − ( probability(t)   − price(t + 72h) )²    
```

Miners have to beat the trivial "Polymarket price stays where it is" baseline at the 72h horizon. The curve points are averaged into a per‑submission score.
