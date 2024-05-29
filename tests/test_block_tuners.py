import pytest
from optuna.pruners import ( 
    SuccessiveHalvingPruner, 
    HyperbandPruner, 
    MedianPruner, 
    NopPruner
)
from optuna.samplers import TPESampler, RandomSampler
from optuna.distributions import ( 
    FloatDistribution, 
    IntDistribution, 
    CategoricalDistribution
)
from sklearn.utils import all_estimators
from itertools import product

from opendesk.utils.synthetic import GeometricBrownianMotion
from opendesk.blocks.tuners import ( 
    WrapSpace, 
    WrapPrune, 
    WrapSearch, 
    TuneConfig,
    TuneEstimator
)


# Getting all regressor estimators from sklearn
estimators = all_estimators(type_filter='regressor')
search_space = WrapSpace()
black_list = [
    # OrthogonalMatchingPursuit need to know n_samples to cast
    # n_nonzero_coefs parameter within the space
    'OrthogonalMatchingPursuit'
]

# Pruner
pruner_options = [
    None,
    WrapPrune('asha'),
    WrapPrune('hyperband'),
    WrapPrune('median')
]
# Optuna pruners
optuna_pruners = [
    ("asha", SuccessiveHalvingPruner),
    ("hyperband", HyperbandPruner),
    ("median", MedianPruner),
    (None, NopPruner),
    (False, NopPruner)
]
# Search spaces
search_alg_options = [
    None,
    WrapSearch('tpe', 123),
    WrapSearch('random', 123)
]
# Optuna search
optuna_samplers = [
    ("tpe", TPESampler),
    ("random", RandomSampler)
]


@pytest.fixture
def data1():
    """Fixture to provide synthetic data with length 5000 and 20 paths."""
    return GeometricBrownianMotion(length=50, num_paths=20).transform()


@pytest.fixture
def data2():
    """Fixture to provide synthetic data with length 5000 and 10 paths."""
    return GeometricBrownianMotion(length=50, num_paths=10).transform()


@pytest.mark.parametrize("name, estimator", all_estimators('regressor'))
def test_wrap_search_space(data1, name, estimator,):
    """
    Test the WrapSearchSpace functionality.

    Parameters
    ----------
    data1 : DataFrame
        The synthetic data to be used for testing.
    """
    # Test when implemented (valid)
    if name in search_space.get_models():
        params = (
            search_space.sample(name)
            if name not in black_list
            else search_space.sample(name, n_samples=len(data1.columns))
        )
        assert isinstance(params, dict), \
            f"WrapSearchSpace {name} does not return a dict."

        for v in params.values():
            assert isinstance(v, (FloatDistribution, IntDistribution, CategoricalDistribution)), \
                f"WrapSearchSpace {name} params is not `optuna` supported."


@pytest.mark.parametrize("pruner_type, pruner_class", optuna_pruners)
def test_pruner(pruner_type, pruner_class):
    """
    Test the creation of different types of pruners.

    Parameters
    ----------
    pruner_type : str or None or False
        The type of pruner to create.
    pruner_class : class
        The expected pruner class.
    """
    pruner = WrapPrune(pruner_type)
    created_pruner = pruner.create_pruner()
    assert isinstance(created_pruner, pruner_class), \
        f"{created_pruner} is not a {pruner_class}"

    if pruner_type not in [None, False]:
        with pytest.raises(ValueError):
            WrapPrune("invalid_pruner").create_pruner()


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
    sampler = WrapSearch(algorithm, seed=42)
    created_sampler = sampler.create_sampler()
    assert isinstance(created_sampler, sampler_class), \
        f"{created_sampler} is not a {sampler_class}"

    with pytest.raises(ValueError):
        WrapSearch("invalid_algorithm", seed=0).create_sampler()


def tune_model(data1, data2, estimator, pruner, search_algorithm, seed):
    # Update config
    config = TuneConfig()
    config.parameter_space = WrapSearch().sample(estimator.__name__, 'optuna')
    config.pruner = pruner
    config.search_algorithm = search_algorithm
    # Tune model
    tuner = TuneEstimator(estimator(), config)
    best_params = tuner.optimize(data1, data2)
    return best_params


@pytest.mark.parametrize("name, estimator", all_estimators('regressor'))
@pytest.mark.parametrize("pruner, search_algorithm", list(product(pruner_options, search_alg_options)))
def test_tune_model(data1, data2, name, estimator, pruner, search_algorithm):
    """
    Test the TuneModel functionality with different estimators.

    Parameters
    ----------
    data1 : DataFrame
        The synthetic data to be used for training.
    data2 : DataFrame
        The synthetic data to be used for testing.
    name : str
        The name of the estimator.
    estimator : class
        The estimator class.
    """
    
    if name in search_space.all_valid_spaces.keys() and name not in black_list:
        best_params = tune_model(
            data1,
            data2,
            estimator,
            pruner,
            search_algorithm,
            43  # seed
        )
        assert isinstance(best_params, dict), \
            f"Output with {pruner}, {search_algorithm} is not a dict"
        assert len(best_params) > 0, \
            "Output is empty"


# Additional specific tests for edge cases or other functionalities can be added here.
