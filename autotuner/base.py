from abc import ABC, abstractmethod
import warnings
from typing import Dict, Any

from optuna import logging as optuna_logger

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit

from autotuner import TuneConfig, WrapSpace, WrapPrune, WrapSearch


class BaseTuner(ABC):
    """
    An abstract base class for building custom tuner classes.

    This class initializes with a configuration object and sets up the
    verbosity, pruning, search algorithm, and parameter space based on that
    configuration.

    Attributes
    ----------
    config : TuneConfig
        Configuration object containing settings for the tuner.
    enable_pruning : bool
        Indicates whether pruning is enabled based on the configuration.
    pruner : WrapPrune
        The pruning strategy to use. Defaults to `None` if pruning is not 
        enabled.
    search_algorithm : WrapSearch
        The search algorithm for exploring the parameter space.
    parameter_space : dict or type
        The parameter space for the tuner to explore.

    Parameters
    ----------
    config : TuneConfig
        The configuration object for the tuner.

    Methods
    -------
    get_param_distributions
        Determines the parameter space for tuning based on the provided 
        configuration.

    """

    @abstractmethod
    def __init__(self, estimator: BaseEstimator | Pipeline, config: TuneConfig):
        self._base_estimator = estimator
        self._config = config
        
        if not self.config.verbose:
            optuna_logger.set_verbosity(optuna_logger.WARNING)
            warnings.filterwarnings('ignore')

    @property
    def base_estimator(self):
        """Compatible sklearn estimator."""
        # Check if estimator adheres to scikit-learn conventions.
        if not isinstance(self._base_estimator, (BaseEstimator, Pipeline)):
            raise ValueError(
                "Estimator does not adhere to scikit-learn conventions: "
                f"{type(self._base_estimator)}"
            )
        return self._base_estimator

    @property
    def config(self):
        """Congiguration dataclass

        More information could be found in autotuner.config.py."""
        return self._config or TuneConfig()

    @property
    def pruner(self):
        """Pruner

        More information could be found in autotuner.wrappers.pruner.py."""
        return (
            self.config.pruner
            if self.config.pruner is not None
            else WrapPrune(None)
        )

    @property
    def search_algorithm(self):
        """Search algorithm

        More information could be found in autotuner.wrappers.search.py."""
        return (
            self.config.search_algorithm or
            WrapSearch("tpe", seed=self.config.random_state)
        )

    @property
    def splitter(self):
        """Cross-validation strategy."""
        return ( 
            TimeSeriesSplit(n_splits=self.config.fold) 
            if self.config.cv is None
            else self.config.cv
        )

        
    def get_param_distributions(self) -> Dict[str, Any] | WrapSpace:
        """
        Set the model parameter space distribution from an estimator or 
        determine the parameter space for the tuner based on the provided 
        configuration.

        The method supports automatic parameter space determination, using 
        predefined types, or directly specifying the space as a dictionary.

        Returns
        -------
        Dict[str, Any] | WrapSpace
            The parameter space to be used for tuning.

        Raises
        ------
        ValueError
            If the parameter space specified in the configuration is of an 
            invalid type.
        """
        if isinstance(self.config.param_space, dict):
            return self.config.param_space

        
        model = (
            self.base_estimator.steps[-1][1] # last step
            if isinstance(self.base_estimator, Pipeline)
            else self.base_estimator
        )
        model_name = model.__class__.__name__

        if self.config.param_space is None or self.config.param_space == 'auto':
            space = WrapSpace.sample(model_name, self.config.search_library)
        
        elif isinstance(self.config.param_space, type):
            space = self.config.param_space.sample(model_name, self.config.search_library)
        
        else:
            raise ValueError("Invalid type for `param_space`.")
            
        # Handle pipeline 
        if isinstance(self.base_estimator, Pipeline):
            for name, step in self.base_estimator.named_steps.items():
                if step == model:
                    prefix = name
            
            space = {f'{prefix}__{key}': value for key, value in space.items()}
        
        return space
            

    @abstractmethod
    def _run_search(self):
        pass
