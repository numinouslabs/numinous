import numpy as np
import pytest

from neurons.validator.utils.common.weights import normalize_max_weight, process_weights


def as_float32(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


class TestNormalizeMaxWeight:
    def test_weights_below_the_limit_are_only_normalised(self):
        normalised = normalize_max_weight(as_float32([1, 1, 2]), limit=0.6)

        np.testing.assert_allclose(normalised, [0.25, 0.25, 0.5], rtol=1e-6)

    def test_a_weight_above_the_limit_is_capped_and_the_rest_spread(self):
        normalised = normalize_max_weight(as_float32([8, 2]), limit=0.6)

        np.testing.assert_allclose(normalised, [0.6, 0.4], atol=1e-5)

    def test_one_dominant_weight_among_many_stays_under_the_limit(self):
        normalised = normalize_max_weight(as_float32([97, 1, 1, 1]), limit=0.4)

        assert normalised.max() <= 0.4 + 1e-5
        assert normalised.sum() == pytest.approx(1.0, abs=1e-6)
        assert normalised.argmax() == 0

    def test_all_zero_weights_become_uniform(self):
        normalised = normalize_max_weight(as_float32([0, 0, 0, 0]), limit=0.5)

        np.testing.assert_allclose(normalised, [0.25] * 4)

    def test_a_limit_too_low_for_the_number_of_weights_gives_uniform_weights(self):
        normalised = normalize_max_weight(as_float32([0.9, 0.1]), limit=0.5)

        np.testing.assert_allclose(normalised, [0.5, 0.5])

    def test_a_single_weight_takes_everything(self):
        np.testing.assert_allclose(normalize_max_weight(as_float32([3.0]), limit=1.0), [1.0])


class TestProcessWeights:
    def test_zero_weights_are_dropped_and_the_rest_normalised(self):
        uids, weights = process_weights(
            uids=np.array([0, 1, 2, 3]),
            weights=as_float32([0, 1, 0, 3]),
            num_neurons=4,
            min_allowed_weights=1,
            max_weight_limit=1.0,
        )

        np.testing.assert_array_equal(uids, [1, 3])
        np.testing.assert_allclose(weights, [0.25, 0.75], rtol=1e-6)

    def test_the_max_weight_limit_is_applied_to_the_uids_with_a_weight(self):
        uids, weights = process_weights(
            uids=np.array([5, 9]),
            weights=as_float32([0.8, 0.2]),
            num_neurons=10,
            min_allowed_weights=1,
            max_weight_limit=0.6,
        )

        np.testing.assert_array_equal(uids, [5, 9])
        np.testing.assert_allclose(weights, [0.6, 0.4], atol=1e-5)

    def test_one_uid_with_all_the_weight_passes_when_the_chain_allows_it(self):
        uids, weights = process_weights(
            uids=np.array([0, 1, 2]),
            weights=as_float32([0, 1, 0]),
            num_neurons=3,
            min_allowed_weights=1,
            max_weight_limit=1.0,
        )

        np.testing.assert_array_equal(uids, [1])
        np.testing.assert_allclose(weights, [1.0])

    def test_all_zero_weights_go_uniform_to_every_neuron(self):
        uids, weights = process_weights(
            uids=np.array([0, 1, 2, 3]),
            weights=as_float32([0, 0, 0, 0]),
            num_neurons=4,
            min_allowed_weights=1,
            max_weight_limit=1.0,
        )

        np.testing.assert_array_equal(uids, [0, 1, 2, 3])
        np.testing.assert_allclose(weights, [0.25] * 4)

    def test_fewer_neurons_than_the_minimum_go_uniform(self):
        uids, weights = process_weights(
            uids=np.array([0, 1]),
            weights=as_float32([1, 0]),
            num_neurons=2,
            min_allowed_weights=3,
            max_weight_limit=1.0,
        )

        np.testing.assert_array_equal(uids, [0, 1])
        np.testing.assert_allclose(weights, [0.5, 0.5])

    def test_fewer_weights_than_the_minimum_pad_every_neuron_by_position(self):
        uids, weights = process_weights(
            uids=np.array([0, 1, 2, 3]),
            weights=as_float32([0, 0, 1, 0]),
            num_neurons=4,
            min_allowed_weights=2,
            max_weight_limit=1.0,
        )

        np.testing.assert_array_equal(uids, [0, 1, 2, 3])
        assert weights.argmax() == 2
        assert weights[2] == pytest.approx(1.0, abs=1e-4)
        assert (weights > 0).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
