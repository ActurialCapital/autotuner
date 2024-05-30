import numpy as np
from sklearn.pipeline import Pipeline
from optuna import distributions


def np_list_arange(
    start: float, stop: float, step: float, inclusive: bool = False
) -> list:
    """
    Generates a list of values within a specified range with step intervals,
    similar to numpy.arange but returns a list. It also ensures correct type 
    handling for floating point values.

    Parameters
    ----------
    start : float
        Starting value of the sequence.
    stop : float
        End value of the sequence.
    step : float
        Step size between consecutive numbers.
    inclusive : bool, optional
        If True, `stop` is the last value in the range, by default False.

    Returns
    -------
    list
        A list of values from `start` to `stop` with `step` intervals.

    Examples
    --------
    >>> print(np_list_arange(1, 5, 1))
    [1, 2, 3, 4]
    >>> print(np_list_arange(1, 5, 1, inclusive=True))
    [1, 2, 3, 4, 5]
    """
    convert_to_float = (
        isinstance(start, float) or isinstance(
            stop, float) or isinstance(step, float)
    )
    if convert_to_float:
        stop = float(stop)
        start = float(start)
        step = float(step)
    stop = stop + (step if inclusive else 0)
    range_ = list(np.arange(start, stop, step))
    range_ = [
        start
        if x < start
        else stop
        if x > stop
        else float(round(x, 15))
        if isinstance(x, float)
        else x
        for x in range_
    ]
    range_[0] = start
    range_[-1] = stop - step
    return range_


def supports_partial_fit(estimator, params: dict = None) -> bool:
    """
    Check if an estimator supports partial_fit.

    Parameters
    ----------
    estimator : object
        The estimator object to check for partial_fit support.

    params : dict, optional
        Additional parameters to check for partial_fit support 
        (default is None).

    Returns
    -------
    bool
        True if the estimator supports partial_fit, False otherwise.

    Notes
    -----
    This function checks if the given estimator supports the partial_fit method,
    which allows for incremental learning. It also considers special cases for 
    certain estimators like MLPClassifier with the lbfgs solver, which does not 
    support partial_fit.

    Examples
    --------
    >>> from sklearn.linear_model import SGDClassifier
    >>> supports_partial_fit(SGDClassifier())
    True

    >>> from sklearn.neural_network import MLPClassifier
    >>> supports_partial_fit(MLPClassifier(solver='lbfgs'))
    False
    """
    # special case for MLP
    from sklearn.neural_network import MLPClassifier

    if isinstance(estimator, MLPClassifier):
        try:
            if (
                params and "solver" in params and "lbfgs" in list(
                    params["solver"])
            ) or estimator.solver == "lbfgs":
                return False
        except Exception:
            return False

    if isinstance(estimator, Pipeline):
        return hasattr(estimator.steps[-1][1], "partial_fit")

    return hasattr(estimator, "partial_fit")


def can_early_stop(
    estimator,
    consider_partial_fit,
    consider_warm_start,
    consider_xgboost,
    params,
):
    """
    Determine if it is possible to perform early stopping during training.

    Parameters
    ----------
    estimator : object
        The estimator object for which early stopping capability is checked.

    consider_partial_fit : bool
        Whether to consider partial_fit as a criterion for early stopping.

    consider_warm_start : bool
        Whether to consider warm_start as a criterion for early stopping.

    consider_xgboost : bool
        Whether to consider XGBoost estimators as a criterion for early 
        stopping.

    params : dict
        Additional parameters to be considered in the decision.

    Returns
    -------
    bool
        True if early stopping is possible, False otherwise.

    Notes
    -----
    This function checks if early stopping is possible for the given estimator 
    based on various criteria, including partial_fit support, warm_start 
    capability, and XGBoost estimators. It is used to determine if early
    stopping can be applied during the training process.

    Examples
    --------
    >>> from sklearn.linear_model import SGDClassifier
    >>> can_early_stop(SGDClassifier(), True, False, False, {})
    True

    >>> from xgboost import XGBClassifier
    >>> can_early_stop(XGBClassifier(), False, True, True, {})
    True
    """
    from sklearn.ensemble import BaseEnsemble
    from sklearn.tree import BaseDecisionTree

    try:
        base_estimator = estimator.steps[-1][1]
    except Exception:
        base_estimator = estimator

    if consider_partial_fit:
        can_partial_fit = supports_partial_fit(base_estimator, params=params)
    else:
        can_partial_fit = False

    if consider_warm_start:
        is_not_tree_subclass = not issubclass(
            type(base_estimator), BaseDecisionTree)
        is_ensemble_subclass = issubclass(type(base_estimator), BaseEnsemble)
        can_warm_start = hasattr(base_estimator, "warm_start") and (
            (
                hasattr(base_estimator, "max_iter")
                and is_not_tree_subclass
                and not is_ensemble_subclass
            )
            or (is_ensemble_subclass and hasattr(base_estimator, "n_estimators"))
        )
    else:
        can_warm_start = False

    is_xgboost = False

    if consider_xgboost:
        from xgboost.sklearn import XGBModel

        is_xgboost = isinstance(base_estimator, XGBModel)

    return can_partial_fit or can_warm_start or is_xgboost


def extract_categorical_from_grid(tune_grid: dict, tune_optuna: dict):
    """ 
    Convert grid search values to categorical distributions if not present
    in 'tune_optuna'.
    """
    for k, v in tune_grid.items():
        if k not in tune_optuna:
            tune_optuna[k] = distributions.CategoricalDistribution(v)


