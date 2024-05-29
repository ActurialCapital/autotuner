from abc import ABC, abstractmethod
from typing import List, Dict, Any
import inspect

from optuna.distributions import (
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution
)

from autotuner import extract_categorical_from_grid, np_list_arange


SearchParams = List[Any]
OptunaDistributions = IntDistribution | FloatDistribution | CategoricalDistribution


class BaseSpace(ABC):
    """
    Base space class designed to define space containers.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    tune_grid : dict of str : list, optional
        The hyperparameters tuning grid for random and grid search, by default 
        an empty dict.
    tune_optuna : dict of str : Distribution, optional
        The hyperparameters tuning grid for optuna, by default an empty dict.

    Attributes
    ----------
    id : str
        ID of the model.
    name : str
        Name of the model.
    tune_grid : dict
        The hyperparameters tuning grid for random and grid search, by default
        an empty dict.
    tune_optuna : dict
        The hyperparameters tuning grid for optuna.

    Examples
    --------
    >>> regressor_space = BaseSearchSpace(
    ...     "rf", 
    ...     "Random Forest",
    ...     {"n_estimators": [100, 200]},
    ...     {"max_depth": CategoricalDistribution([3, 5, 7])}
    ... )
    >>> print(regressor_space.id)
    'rf'
    >>> print(regressor_space.name)
    'Random Forest'
    """

    @abstractmethod
    def __init__(
        self,
        id: str,
        name: str,
        tune_grid: Dict[str, SearchParams],
        tune_optuna: Dict[str, OptunaDistributions]
    ):
        self.id = id
        self.name = name
        self.tune_grid = tune_grid
        self.tune_optuna = tune_optuna

    def __repr__(self):
        """
        Special method to return the string representation of the instance,
        dynamically using the class name of the subclass.
        """
        return f"{self.__class__.__name__}"

    def get_params(self) -> dict:
        """
        Returns a dictionary representation of the model space to be optimized.

        Returns
        -------
        dict
            A dictionary containing the model's ID, name, tuning grid search,
            and tuning optuna.

        """
        return {
            "id": self.id,
            "name": self.name,
            "grid": self.tune_grid,
            "optuna": self.tune_optuna
        }


class LinearRegression(BaseSpace):
    """
    Linear Regression Hyperparameter Search Space.

    This class defines the hyperparameter search space for Linear Regression 
    models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "lr"
        tune_grid = {"fit_intercept": [True, False]}
        tune_optuna = {}
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Linear Regression",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,
        )


class GaussianProcessRegressor(BaseSpace):
    """
    Gaussian Process Hyperparameter Search Space.

    This class defines the hyperparameter search space for Gaussian Process 
    Regression models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "gauss"
        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
        }
        tune_optuna = {"alpha": FloatDistribution(0.001, 10)}
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Gaussian Process",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,
        )


class Lasso(BaseSpace):
    """
    Lasso Regression Hyperparameter Search Space.

    This class defines the hyperparameter search space for Lasso Regression 
    models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "lasso"
        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "fit_intercept": [True, False],
        }
        tune_optuna = {"alpha": FloatDistribution(0.001, 10)}
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Lasso Regression",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,
        )


class Ridge(BaseSpace):
    """
    Ridge Regression Hyperparameter Search Space.

    This class defines the hyperparameter search space for Ridge Regression 
    models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "ridge"
        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "fit_intercept": [True, False],
        }
        tune_optuna = {"alpha": FloatDistribution(0.001, 10)}
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Ridge Regression",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class RidgeCV(BaseSpace):
    """
    Ridge CV Regression Hyperparameter Search Space.

    This class defines the hyperparameter search space for Ridge CV Regression 
    models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "ridge_cv"
        tune_grid = {
            "alphas": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "fit_intercept": [True, False],
        }
        tune_optuna = {
            "alphas": FloatDistribution(0.001, 10)}
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Ridge CV Regression",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class ElasticNet(BaseSpace):
    """
    Elastic Net Hyperparameter Search Space.

    This class defines the hyperparameter search space for Elastic Net models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "en"
        tune_grid = {
            "alpha": np_list_arange(
                0.01, 10, 0.01, inclusive=True),
            "l1_ratio": np_list_arange(
                0.01, 1, 0.001, inclusive=False),
            "fit_intercept": [True, False],
        }
        tune_optuna = {
            "alpha": FloatDistribution(
                0, 1),
            "l1_ratio": FloatDistribution(
                0.01, 0.9999999999),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Elastic Net",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class Lars(BaseSpace):
    """
    Least Angle Regression Hyperparameter Search Space.

    This class defines the hyperparameter search space for Least Angle Regression 
    models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "lar",
        tune_grid = {
            "fit_intercept": [True, False],
            "eps": [
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.05,
                0.0005,
                0.005,
                0.00005,
                0.02,
                0.007,
                0.1,
            ],
        }
        tune_optuna = {
            "eps": FloatDistribution(
                0.00001, 0.1),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Least Angle Regression",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class LassoLars(BaseSpace):
    """
    Lasso Least Angle Regression Hyperparameter Search Space.

    This class defines the hyperparameter search space for Lasso Least Angle 
    Regression models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "llar"
        tune_grid = {
            "fit_intercept": [True, False],
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
            "eps": [
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.05,
                0.0005,
                0.005,
                0.00005,
                0.02,
                0.007,
                0.1,
            ],
        }
        tune_optuna = {
            "eps": FloatDistribution(
                0.00001, 0.1),
            "alpha": FloatDistribution(
                0.0000000001, 0.9999999999, log=True),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Lasso Least Angle Regression",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class OrthogonalMatchingPursuit(BaseSpace):
    """
    Orthogonal Matching Pursuit Hyperparameter Search Space.

    This class defines the hyperparameter search space for Orthogonal Matching 
    Pursuit models.

    Args
    ----------
    n_samples : int
        Number of samples in the dataset.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids for grid search.
    tune_optuna : dict
        Dictionary of hyperparameter distributions for Optuna.

    """

    def __init__(self, n_samples: int):
        id = "omp"
        tune_grid = {
            "n_nonzero_coefs": range(1, n_samples + 1),
            "fit_intercept": [True, False],
        }
        tune_optuna = {
            "n_nonzero_coefs": IntDistribution(1, n_samples)
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Orthogonal Matching Pursuit",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class KernelRidge(BaseSpace):
    """
    Kernel Ridge Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for Kernel Ridge 
    Regressor models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "kr"
        tune_grid = {
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
        }
        tune_optuna = {
            "alpha": FloatDistribution(
                0.0000000001,
                0.9999999999,
                log=True
            ),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Kernel Ridge",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class KNeighborsRegressor(BaseSpace):
    """
    K Neighbors Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for K Neighbors Regressor models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "knn"
        tune_grid = {}
        tune_optuna = {}
        tune_grid["n_neighbors"] = range(1, 51)
        tune_grid["weights"] = ["uniform", "distance"]
        tune_grid["metric"] = ["minkowski", "euclidean", "manhattan"]
        tune_optuna["n_neighbors"] = IntDistribution(1, 51)
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="K Neighbors Regressor",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class DecisionTreeRegressor(BaseSpace):
    """
    Decision Tree Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for Decision Tree Regressor models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "dt"
        tune_grid = {
            "max_depth": np_list_arange(1, 16, 1, inclusive=True),
            "max_features": [1.0, "sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "criterion": ["squared_error", "absolute_error", "friedman_mse"],
        }
        tune_optuna = {
            "max_depth": IntDistribution(
                1, 16),
            "max_features": FloatDistribution(
                0.4, 1),
            "min_samples_leaf": IntDistribution(
                2, 6),
            "min_samples_split": IntDistribution(
                2, 10),
            "min_impurity_decrease": FloatDistribution(
                0.000000001, 0.5, log=True),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Decision Tree Regressor",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class RandomForestRegressor(BaseSpace):
    """
    Random Forest Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for Random Forest Regressor models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "rf"
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        tune_grid["criterion"] = ["squared_error", "absolute_error"]
        tune_grid["min_samples_split"] = [
            2, 5, 7, 9, 10]
        tune_grid["min_samples_leaf"] = [
            2, 3, 4, 5, 6]
        tune_optuna = {
            "n_estimators": IntDistribution(
                10, 300),
            "max_depth": IntDistribution(
                1, 11),
            "min_impurity_decrease": FloatDistribution(
                0.000000001, 0.5, log=True),
            "max_features": FloatDistribution(
                0.4, 1),
        }
        tune_optuna["min_samples_split"] = IntDistribution(
            2, 10)
        tune_optuna["min_samples_leaf"] = IntDistribution(
            2, 6)
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Random Forest Regressor",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class ExtraTreeRegressor(BaseSpace):
    """
    Extra Tree Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for Extra Tree Regressor models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.
    """

    def __init__(self, **kwargs):
        id = "et"
        tune_grid = {
            "criterion": ["squared_error", "absolute_error"],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf": [2, 3, 4, 5, 6],
        }
        tune_optuna = {
            "max_depth": IntDistribution(
                1, 11),
            "min_samples_split": IntDistribution(
                2, 10),
            "min_samples_leaf": IntDistribution(
                1, 5),
            "max_features": FloatDistribution(
                0.4, 1),
            "min_impurity_decrease": FloatDistribution(
                0.000000001, 0.5, log=True),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Extra Tree Regressor",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,
        )


class ExtraTreesRegressor(BaseSpace):
    """
    Extra Trees Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for Extra Trees 
    Regressor models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.
    """

    def __init__(self, **kwargs):
        id = "ets"
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "criterion": ["squared_error", "absolute_error"],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf": [2, 3, 4, 5, 6],
        }
        tune_optuna = {
            "n_estimators": IntDistribution(
                10, 300),
            "max_depth": IntDistribution(
                1, 11),
            "min_samples_split": IntDistribution(
                2, 10),
            "min_samples_leaf": IntDistribution(
                1, 5),
            "max_features": FloatDistribution(
                0.4, 1),
            "min_impurity_decrease": FloatDistribution(
                0.000000001, 0.5, log=True),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Extra Trees Regressor",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,
        )


class MLPRegressor(BaseSpace):
    """
    MLP Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for MLP Regressor models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    Notes
    -----
    MLPRegressor trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E.
        "Connectionist learning procedures." Artificial intelligence 40.1
        (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks." International Conference
        on Artificial Intelligence and Statistics. 2010.

    He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).

    Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic
        optimization." arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(self, **kwargs):
        id = "mlp"
        tune_grid = {
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
            "hidden_layer_sizes": [10, 200],
            "activation": ["tanh", "identity", "logistic", "relu"],
            "solver": ['lbfgs', 'sgd', 'adam'],
            "batch_size": ['auto', 32, 64, 128]
        }
        tune_optuna = {
            "alpha": FloatDistribution(
                0.0000000001, 0.9999999999, log=True),
            "hidden_layer_sizes": IntDistribution(
                10, 200),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="MLP Regressor",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class XGBRegressor(BaseSpace):
    """
    Extreme Gradient Boosting Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for Extreme Gradient Boosting Regressor models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "xgboost"
        tune_grid = {
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "subsample": [0.2, 0.3, 0.5, 0.7, 0.9, 1],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "min_child_weight": [1, 2, 3, 4],
            "reg_alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "scale_pos_weight": np_list_arange(0, 50, 0.1, inclusive=True),
        }
        tune_optuna = {
            "learning_rate": FloatDistribution(
                0.000001, 0.5, log=True),
            "n_estimators": IntDistribution(
                10, 300),
            "subsample": FloatDistribution(
                0.2, 1),
            "max_depth": IntDistribution(
                1, 11),
            "colsample_bytree": FloatDistribution(
                0.5, 1),
            "min_child_weight": IntDistribution(
                1, 4),
            "reg_alpha": FloatDistribution(
                0.0000000001, 10, log=True),
            "reg_lambda": FloatDistribution(
                0.0000000001, 10, log=True),
            "scale_pos_weight": FloatDistribution(
                1, 50),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Extreme Gradient Boosting",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,

        )


class BaggingRegressor(BaseSpace):
    """
    Bagging Regressor Hyperparameter Search Space.

    This class defines the hyperparameter search space for Bagging Regressor 
    models.

    Attributes
    ----------
    id : str
        Identifier for the search space.
    name : str
        Name of the regression model.
    tune_grid : dict
        Dictionary of hyperparameter grids.
    tune_optuna : dict
        Dictionary of hyperparameter distributions.

    """

    def __init__(self, **kwargs):
        id = "Bagging"
        tune_grid = {
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
            "max_features": np_list_arange(0.4, 1, 0.1, inclusive=True),
            "max_samples": np_list_arange(0.4, 1, 0.1, inclusive=True),
        }
        tune_optuna = {
            "max_features": FloatDistribution(0.4, 1),
            "max_samples": FloatDistribution(0.4, 1),
        }
        extract_categorical_from_grid(tune_grid, tune_optuna)
        BaseSpace.__init__(
            self,
            id=id,
            name="Bagging Regressor",
            tune_grid=tune_grid,
            tune_optuna=tune_optuna,
        )


class WrapSpace:
    """
    A class responsible for sampling hyperparameters for a given estimator 
    using predefined hyperparameter spaces.

    It uses the estimator name to find and instantiate the appropriate 
    hyperparameter space and samples parameters. Therefore, both estimator and 
    estimator space classes should have the same name.

    Attributes
    ----------
    get_available_models : List[str]
        Lists all available estimators.

    Methods
    -------
    sample(estimator_name, search_library)
        Samples parameters for the specified estimator.

    """

    @classmethod
    def get_models(cls) -> List[str]:
        """ 
        Lists all available estimators.

        Returns
        -------
        List[str]
            List of available estimators.

        Examples
        --------
        ```pycon
        >>> WrapSpace.get_models()
        ['BaggingRegressor',
         'DecisionTreeRegressor',
         'ElasticNet',
         'ExtraTreeRegressor',
         'ExtraTreesRegressor',
         'GaussianProcessRegressor',
         'KNeighborsRegressor',
         'KernelRidge',
         'Lars',
         'Lasso',
         'LassoLars',
         'LinearRegression',
         'MLPRegressor',
         'OrthogonalMatchingPursuit',
         'RandomForestRegressor',
         'Ridge',
         'RidgeCV',
         'XGBRegressor']
        ```

        """
        module = inspect.getmodule(cls)
        exclude = ['WrapSpace', 'self']
        return [
            name for name, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == module.__name__ and name not in exclude
        ]

    @classmethod
    def sample(
        cls,
        estimator_name: str,
        search_library: str = 'optuna',
        **search_space_kwargs
    ) -> Dict[str, SearchParams | OptunaDistributions]:
        """
        Samples parameters for a given estimator based on the predefined 
        hyperparameter space.

        Parameters
        ----------
        estimator_name : str
            The name of the estimator for which to sample parameters.
        search_library : str
            The library to use for search optimisation.
        search_space_kwargs : Any
            Key word extra arguments in search space. For example, 
            `OrthogonalMatchingPursuit` requires `n_samples`.

        Returns
        -------
        Dict[str, List[Any] | distributions]
            A dictionary of sampled parameters for the given estimator.

        Raises
        ------
        ValueError
            If the estimator name is not integrated.

        ValueError
            if search library is not supported.

        Examples
        --------
        Using Scikit-Learn `LinearRegression`
        ```pycon
        >>> from sklearn.linear_model import LinearRegression
        >>> WrapSpace.sample('LinearRegression')
        {'fit_intercept': CategoricalDistribution(choices=(True, False))}
        ```

        Using Scikit-Learn `RandomForestRegressor`
        ```pycon
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> WrapSpace.sample('RandomForestRegressor')
        {'n_estimators': IntDistribution(high=300, log=False, low=10, step=1),
         'max_depth': IntDistribution(high=11, log=False, low=1, step=1),
         'min_impurity_decrease': FloatDistribution(high=0.5, log=True, low=1e-09, step=None),
         'max_features': FloatDistribution(high=1.0, log=False, low=0.4, step=None),
         'min_samples_split': IntDistribution(high=10, log=False, low=2, step=1),
         'min_samples_leaf': IntDistribution(high=6, log=False, low=2, step=1),
         'bootstrap': CategoricalDistribution(choices=(True, False)),
         'criterion': CategoricalDistribution(choices=('squared_error', 'absolute_error'))}
        ```  

        Using Scikit-Learn `MLPRegressor`
        ```pycon
        >>> from sklearn.neural_network import MLPRegressor
        >>> WrapSpace.sample('MLPRegressor')
        {'alpha': FloatDistribution(high=0.9999999999, log=True, low=1e-10, step=None),
         'hidden_layer_sizes': IntDistribution(high=200, log=False, low=10, step=1),
         'learning_rate': CategoricalDistribution(choices=('constant', 'invscaling', 'adaptive')),
         'activation': CategoricalDistribution(choices=('tanh', 'identity', 'logistic', 'relu')),
         'solver': CategoricalDistribution(choices=('lbfgs', 'sgd', 'adam')),
         'batch_size': CategoricalDistribution(choices=('auto', 32, 64, 128))}
        ```

        """
        try:
            module = inspect.getmodule(cls)
            space_class = getattr(module, estimator_name)
        except:
            raise ValueError(f"Estimator {estimator_name} is not integrated.")
        
        # Grid search/random are WIP
        if search_library not in ['optuna']:
            raise ValueError(
                f"Search library {search_library} is not supported."
            )
            
        return space_class(**search_space_kwargs).get_params()[search_library]
