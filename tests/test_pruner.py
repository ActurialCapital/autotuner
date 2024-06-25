import pytest

from optuna.pruners import (
    SuccessiveHalvingPruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner
)

from autotuner import WrapPrune


# Optuna pruners
optuna_pruners = [
    ("asha", SuccessiveHalvingPruner),
    ("hyperband", HyperbandPruner),
    ("median", MedianPruner),
    (None, NopPruner),
    (False, NopPruner)
]


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
        with pytest.raises(AttributeError):
            WrapPrune("invalid_pruner").create_pruner()
