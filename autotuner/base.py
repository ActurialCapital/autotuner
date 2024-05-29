from abc import ABC, abstractmethod
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from autotuner import TuneConfig, WrapSpace


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
    def __init__(
        self,
        estimator: BaseEstimator | Pipeline,
        config: TuneConfig
    ):
        self.estimator = estimator
        self.config = config or TuneConfig()

    def get_param_distributions(
        self,
        estimator: BaseEstimator | Pipeline = None
    ) -> Dict[str, Any] | WrapSpace:
        """
        Set the model parameter space distribution from an estimator or 
        determine the parameter space for the tuner based on the provided 
        configuration.

        The method supports automatic parameter space determination by passing
        the estimator (optional), using predefined types, or directly 
        specifying the space as a dictionary.

        Parameters
        ----------
        estimator : BaseEstimator, optional
            Base estimator instance.

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

        if isinstance(estimator, Pipeline):
            model_name = estimator.steps[-1][1].__class__.__name__
        else:
            model_name = estimator.__class__.__name__

        if self.config.param_space is None or self.config.param_space == 'auto':
            return WrapSpace.sample(
                model_name,
                self.config.search_library
            )
        elif isinstance(self.config.param_space, type):
            return self.config.param_space.sample(
                model_name,
                self.config.search_library
            )
        else:
            raise ValueError("Invalid type for `param_space`.")

    @abstractmethod
    def _run_search(self):
        pass
