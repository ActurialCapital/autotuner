from dataclasses import dataclass
from typing import Dict, Any
import random
from sklearn.model_selection import BaseCrossValidator
from optuna.samplers import BaseSampler

from autotuner import WrapSpace, WrapSearch, WrapPrune


@dataclass
class TuneConfig:
    """
    Configuration class for tuning machine learning models.

    This class defines the configuration parameters for hyperparameter tuning
    of machine learning models using Optuna or other search libraries.

    Attributes
    ----------
    param_space : WrapSearchSpace | dict, optional
        The hyperparameter space to search. Can be a custom search space
        or a dictionary of parameter distributions.
    pruner : WrapPrune, optional
        The pruner used for Optuna's study.
    search_algorithm : WrapSearch, optional
        The sampler used for Optuna's study.
    scoring : str, default "neg_mean_squared_error"
        The scoring metric used for optimization.
    cv : BaseCrossValidator, default to None
        Cross-validator. if None, model uses `TimeSeriesSplit` with fold.
    fold : int, default 10
        The number of folds for cross-validation.
    n_trials : int, default 10
        The number of trials for hyperparameter search.
    early_stopping_max_iters : int, default 10
        The maximum number of iterations for early stopping.
    return_train_score : bool, default False
        Flag indicating whether to return the training score.
    search_library : str, default 'optuna'
        The library used for hyperparameter search.
    verbose : bool, default False
        Verbosity flag for logging.
    random_state : int, optional
        The random state seed for reproducibility. If not provided, a random
        seed between 150 and 9000 is chosen.

    """
    study_name: str = "autotuner"
    param_space: WrapSpace | Dict[str, Any] = None
    pruner: WrapPrune = None
    search_algorithm: WrapSearch | BaseSampler = None
    scoring: str = "neg_mean_squared_error"
    direction: str = "maximize"
    cv: BaseCrossValidator = None
    fold: int = 10
    n_trials: int = 10
    early_stopping_max_iters: int = 10
    return_train_score: bool = False
    search_library: str = 'optuna'
    verbose: bool = False
    random_state: int = None

    def __post_init__(self):
        if self.random_state is None:
            self.random_state = random.randint(150, 9000)
