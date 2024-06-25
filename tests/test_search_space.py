import pytest
import numpy as np

from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution
)
from sklearn.utils import all_estimators
from autotuner import WrapSpace

# Data
length = 50
paths = 20

# Getting all regressor estimators from sklearn
sk_estimators = all_estimators(type_filter='regressor')

# Special signature
black_list = [
    # OrthogonalMatchingPursuit need to know n_samples to cast
    # n_nonzero_coefs parameter within the space
    'OrthogonalMatchingPursuit'
]

# Search space
search_space = WrapSpace()

@pytest.fixture
def X_train() -> np.ndarray:
    """Fixture to provide synthetic data with length 50 and 20 paths."""
    return np.random.normal(size=(length, paths))


@pytest.mark.parametrize("name, estimator", sk_estimators)
def test_wrap_search_space(X_train, name, estimator):
    """Test the WrapSearchSpace functionality."""
    if name in search_space.get_models():
        params = (
            search_space.sample(name)
            if name not in black_list
            else search_space.sample(name, n_samples=length)
        )
        assert isinstance(params, dict), \
            f"WrapSearch {name} does not return a dict."

        for value in params.values():
            assert isinstance(
                value,
                (FloatDistribution, IntDistribution, CategoricalDistribution)
            ), f"Search space {name} params is not `optuna` supported."
