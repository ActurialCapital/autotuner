import pytest
from itertools import product

import numpy as np
from sklearn.utils import all_estimators

from autotuner import WrapSpace, WrapPrune, WrapSearch, TuneConfig, TuneEstimator


seed = 4560

# Data
length = 50
paths = 20

# Getting all regressor estimators from sklearn
sk_estimators = all_estimators(type_filter='regressor')

# Special signature
black_list = [
    # OrthogonalMatchingPursuit need to know n_samples to cast
    # n_nonzero_coefs parameter within the space
    'OrthogonalMatchingPursuit',
    # Time consuming - tested independently
    'ExtraTreesRegressor',
    'Lars',
    'LassoLars',
    'MLPRegressor',
    'RandomForestRegressor'
]

# Search space
search_space = WrapSpace()

# Pruner
pruner_options = [None, False, 'asha', 'hyperband', 'median']

# Search
search_alg_options = [None, False, 'tpe', 'random']


@pytest.fixture
def X_train() -> np.ndarray:
    """Fixture to provide synthetic data with length 50 and 20 paths."""
    return np.random.normal(size=(length, paths))


@pytest.fixture
def y_train() -> np.ndarray:
    """Fixture to provide synthetic data with length 50 and 10 paths."""
    return np.random.normal(size=(length, paths))


def get_best_params(X_train, y_train, estimator, pruner, search_algorithm, seed):
    # Update config
    config = TuneConfig()
    config.param_space = 'auto'
    config.pruner = WrapPrune(pruner)
    config.search_algorithm = WrapSearch(search_algorithm, seed)

    # Fit Tuned model
    model = TuneEstimator(estimator(), config).fit(X_train, y_train)
    return model.best_params_


@pytest.mark.parametrize("name, estimator", sk_estimators)
@pytest.mark.parametrize("pruner, search_algorithm", list(product(pruner_options, search_alg_options)))
def test_tune_model(
    X_train,
    y_train,
    name,
    estimator,
    pruner,
    search_algorithm
):
    """
    Test the TuneModel functionality with different estimators.

    Parameters
    ----------
    X_train : np.ndarray
        The synthetic data to be used for training.
    y_train : np.ndarray
        The synthetic data to be used for testing.
    name : str
        The name of the estimator.
    estimator : class
        The estimator class.
    """
    for name, estimator in sk_estimators:
        for pruner, search_algorithm in list(product(pruner_options, search_alg_options)):
                        
            if name in search_space.get_models() and name not in black_list:
                best_params = get_best_params(
                    X_train,
                    y_train,
                    estimator,
                    pruner,
                    search_algorithm,
                    seed=seed
                )
                print(name, best_params)
                assert isinstance(best_params, dict), \
                    f"Output with {pruner}, {search_algorithm} is not a dict"
                assert len(best_params) > 0, "Output `best_params_` is empty"
