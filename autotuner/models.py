import warnings

from optuna import create_study as create_optuna_study
from optuna.integration import OptunaSearchCV
from optuna import logging as optuna_logger

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from autotuner import BaseTuner, TuneConfig, WrapPrune, WrapSearch, can_early_stop


class TuneEstimator(BaseTuner, OptunaSearchCV):
    """    
    Tuner for `sklearn` machine learning estimators using Optuna.

    Extends `BaseTuner` to provide functionality for tuning hyperparameters of 
    machine learning models. Utilizes Optuna `OptunaSearchCV` for optimization, 
    supporting various parameter samplers, pruning strategies, and the 
    capability to handle cross-validation.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        The machine learning estimator to tune.
    config : TunerConfig
        The configuration object for tuning.

    Methods
    -------
    set_verbosity
        Static method to set the verbosity level for Optuna's logger based on 
        the provided flag.

    """

    def __init__(
        self, 
        estimator: BaseEstimator | Pipeline,
        config: TuneConfig = None
    ):
        BaseTuner.__init__(self, estimator, config)
        self.set_verbosity(self.config.verbose)
        self.enable_pruning = self.config.pruner is not None
        self.pruner = (
            self.config.pruner
            if self.enable_pruning
            else WrapPrune(None)
        )
        self.search_algorithm = (
            self.config.search_algorithm or WrapSearch(
                "tpe",
                seed=self.config.random_state
            )
        )
        self._run_search()

    @staticmethod
    def set_verbosity(verbose: bool):
        """
        Set the verbosity level for Optuna's logger.

        This is a static method. If `verbose` is False, it sets Optuna's 
        logger to WARNING level and ignores warnings.

        Parameters
        ----------
        verbose : bool
            If True, detailed logging is enabled. If False, minimal logging is 
            used.
        """
        if not verbose:
            optuna_logger.set_verbosity(optuna_logger.WARNING)
            warnings.filterwarnings('ignore')

    def _run_search(self):
        """
        Tunes the hyperparameters of the specified estimator using Optuna.

        Sets up an Optuna study and conducts hyperparameter tuning with 
        cross-validation.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features for the model.
        y : pd.Series or np.ndarray
            Target values corresponding to the input features.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the best hyperparameters found during the 
            tuning process.

        Examples
        --------
        Run a Random Forest Regressor
        ```pycon
        >>> from opendesk.blocks.tuners import TuneEstimator
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> tuner = TuneEstimator(RandomForestRegressor())
        >>> tuner.fit(X, y)
        TuneEstimator(
            estimator=RandomForestRegressor(),
            config=TuneConfig(
                study_name='opendesk',
                param_space=None,
                pruner=None,
                search_algorithm=None,
                scoring='neg_mean_squared_error',
                direction='maximize',
                fold=10,
                n_trials=10,
                early_stopping_max_iters=10,
                return_train_score=False,
                search_library='optuna',
                verbose=False,
                random_state=8101)
            
        )
        >>> tuner.best_params_
        {'n_estimators': 95,
         'max_depth': 5,
         'min_impurity_decrease': 5.099297290041241e-09,
         'max_features': 0.5535777014634076,
         'min_samples_split': 6,
         'min_samples_leaf': 4,
         'bootstrap': False,
         'criterion': 'squared_error'}
        ```
        """
        study = create_optuna_study(
            direction=self.config.direction,
            sampler=self.search_algorithm.create_sampler(),
            pruner=self.pruner.create_pruner(),
            study_name=self.config.study_name
        )
        OptunaSearchCV.__init__(
            self,
            estimator=self.estimator,
            param_distributions=self.get_param_distributions(self.estimator),
            cv=TimeSeriesSplit(n_splits=self.config.fold),
            enable_pruning=self.enable_pruning and can_early_stop(
                self.estimator, True, False, False, self.parameter_space
            ),
            n_trials=self.config.n_trials,
            scoring=self.config.scoring,
            study=study,
            refit=True,
            return_train_score=self.config.return_train_score,
            verbose=self.config.verbose,
            random_state=self.config.random_state,
            error_score="raise"
        )
