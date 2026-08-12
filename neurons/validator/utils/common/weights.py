import numpy as np
from numpy.typing import NDArray

U16_MAX = 65535


def normalize_max_weight(weights: NDArray[np.float32], limit: float) -> NDArray[np.float32]:
    epsilon = 1e-7

    normalized = weights.copy()
    values = np.sort(normalized)

    if weights.sum() == 0 or weights.shape[0] * limit <= 1:
        return np.ones_like(weights) / weights.shape[0]

    estimation = values / values.sum()

    if estimation.max() <= limit:
        return normalized / normalized.sum()

    cumsum = np.cumsum(estimation, 0)

    estimation_sum = np.array(
        [(len(values) - index - 1) * estimation[index] for index in range(len(values))]
    )
    n_values = (estimation / (estimation_sum + cumsum + epsilon) < limit).sum()

    cutoff_scale = (limit * cumsum[n_values - 1] - epsilon) / (
        1 - (limit * (len(estimation) - n_values))
    )
    cutoff = cutoff_scale * values.sum()

    normalized[normalized > cutoff] = cutoff

    return normalized / normalized.sum()


def process_weights(
    uids: NDArray[np.int64],
    weights: NDArray[np.float32],
    num_neurons: int,
    min_allowed_weights: int,
    max_weight_limit: float,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    if not isinstance(weights, np.float32):
        weights = weights.astype(np.float32)

    non_zero_weight_idx = np.argwhere(weights > 0).squeeze(axis=1)
    non_zero_weight_uids = uids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]

    if non_zero_weights.size == 0 or num_neurons < min_allowed_weights:
        final_weights = np.ones(num_neurons, dtype=np.int64) / num_neurons

        return np.arange(len(final_weights)), final_weights

    if non_zero_weights.size < min_allowed_weights:
        weights = np.ones(num_neurons, dtype=np.int64) * 1e-5
        weights[non_zero_weight_idx] += non_zero_weights
        normalized_weights = normalize_max_weight(weights=weights, limit=max_weight_limit)

        return np.arange(len(normalized_weights)), normalized_weights

    normalized_weights = normalize_max_weight(weights=non_zero_weights, limit=max_weight_limit)

    return non_zero_weight_uids, normalized_weights
