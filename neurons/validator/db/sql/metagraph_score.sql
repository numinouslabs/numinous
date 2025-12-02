WITH reference_event AS (
    -- Get the reference ROWID for the specified event.
    SELECT MIN(ROWID) AS reference_row
    FROM scores
    WHERE event_id = :event_id
),
all_events AS (
    -- For each event, get the smallest ROWID (as a proxy for event order).
    SELECT event_id, MIN(ROWID) AS event_min_row
    FROM scores
    GROUP BY event_id
),
last_n_events AS (
    -- Select the last N events that occurred before the reference event, including the reference event.
    SELECT event_id
    FROM all_events
    WHERE event_min_row <= (SELECT reference_row FROM reference_event)
    ORDER BY event_min_row DESC
    LIMIT :n_events
),
miner_event_briers AS (
    -- Calculate Brier score for each miner's prediction for each event in the window
    SELECT
        s.miner_uid,
        s.miner_hotkey,
        s.event_id,
        s.prediction,
        s.event_score AS brier_score
    FROM scores s
    JOIN events e ON s.event_id = e.event_id
    WHERE s.event_id IN (SELECT event_id FROM last_n_events)
),
avg_brier_per_miner AS (
    -- Compute the average Brier score for each miner over the last N events
    SELECT
        miner_uid,
        miner_hotkey,
        AVG(brier_score) AS average_brier_score
    FROM miner_event_briers
    GROUP BY miner_uid, miner_hotkey
),
ranked_miners AS (
    -- Rank miners by average Brier score (lowest is best), with miner_uid as tie-breaker
    SELECT
        miner_uid,
        miner_hotkey,
        average_brier_score,
        ROW_NUMBER() OVER (ORDER BY average_brier_score ASC, miner_uid ASC) as rank
    FROM avg_brier_per_miner
),
power_rank_sum AS (
    -- Calculate sum of power-adjusted ranks for non-winners (for normalization)
    SELECT SUM(POWER(rank, -:decay_power)) as total_power_rank
    FROM ranked_miners
    WHERE miner_uid != :burn_uid AND rank > 1
),
payload AS (
    -- Assign metagraph score: burn_weight to burn UID, winner_weight to rank 1, remainder distributed by power rank
    SELECT
        rm.miner_uid,
        rm.miner_hotkey,
        CASE
            WHEN rm.miner_uid = :burn_uid THEN :burn_weight
            WHEN rm.rank = 1 THEN (1.0 - :burn_weight) * :winner_weight
            ELSE (
                (1.0 - :burn_weight) * (1.0 - :winner_weight)
                * POWER(rm.rank, -:decay_power)
                / (SELECT total_power_rank FROM power_rank_sum)
            )
        END AS metagraph_score,
        json_object(
            'average_brier_score', rm.average_brier_score,
            'rank', rm.rank
        ) AS other_data
    FROM ranked_miners rm
)
UPDATE scores
SET
    metagraph_score = (
        SELECT metagraph_score FROM payload
        WHERE miner_uid = scores.miner_uid
        AND miner_hotkey = scores.miner_hotkey
    ),
    other_data = (
        SELECT other_data FROM payload
        WHERE miner_uid = scores.miner_uid
        AND miner_hotkey = scores.miner_hotkey
    ),
    processed = 1
WHERE event_id = :event_id
;