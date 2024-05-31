from optuna import create_study as create_optuna_study
from optuna.integration import OptunaSearchCV

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from autotuner import BaseTuner, TuneConfig, can_early_stop


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

    Examples
    --------
    Run a Random Forest Regressor
    ```pycon
    >>> from autotuner import TuneEstimator
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = TuneEstimator(RandomForestRegressor())
    >>> model.fit(X, y)
    TuneEstimator(
        estimator=RandomForestRegressor(),
        config=TuneConfig(
            study_name='opendesk',
            param_space=None,
            pruner=None,
            search_algorithm=None,
            scoring='neg_mean_squared_error',
            direction='maximize',
            cv=None,
            fold=10,
            n_trials=10,
            early_stopping_max_iters=10,
            return_train_score=False,
            search_library='optuna',
            verbose=False,
            random_state=8101)

    )
    >>> model.best_params_
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

    def __init__(self, estimator: BaseEstimator | Pipeline, config: TuneConfig = None):
        BaseTuner.__init__(self, estimator, config)
        self._run_search()

    def _run_search(self):
        """
        Tunes the hyperparameters of the specified `sklearn` estimator using 
        Optuna.

        Sets up an Optuna study and search all candidates in grid with 
        cross-validation.

        """
        study = create_optuna_study(
            direction=self.config.direction,
            sampler=self.search_algorithm.create_sampler(),
            pruner=self.pruner.create_pruner(),
            study_name=self.config.study_name
        )
        OptunaSearchCV.__init__(
            self,
            estimator=self.base_estimator,
            param_distributions=self.get_param_distributions(),
            cv=self.splitter,
            enable_pruning=self.config.pruner is not None and can_early_stop(
                self.base_estimator, True, False, False, self.parameter_space
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
