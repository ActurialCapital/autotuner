import pytest

from optuna.samplers import (
    TPESampler,
    RandomSampler
)

from autotuner import WrapSearch


seed = 4560

# Optuna search
optuna_samplers = [
    ("tpe", TPESampler),
    ("random", RandomSampler)
]


@pytest.mark.parametrize("algorithm, sampler_class", optuna_samplers)
def test_search_algorithm(algorithm, sampler_class):
    """
    Test the creation of different search algorithms.

    Parameters
    ----------
    algorithm : str
        The type of search algorithm to create.
    sampler_class : class
        The expected sampler class.
    """
    sampler = WrapSearch(algorithm, seed=seed)
    created_sampler = sampler.create_sampler()
    assert isinstance(created_sampler, sampler_class), \
        f"{created_sampler} is not a {sampler_class}"

    with pytest.raises((ValueError, AttributeError)):
        WrapSearch("invalid_algorithm", seed=seed).create_sampler()
