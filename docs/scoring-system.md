# Scoring System

This document explains the domain-ranking based scoring and payout system shown on the [Numinous leaderboard](https://leaderboard.numinouslabs.io/miner-console/setup?step=-1&coldkey=&hotkey=&browse=false).

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

