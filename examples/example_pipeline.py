from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from autotuner import TuneEstimator


if __name__ == "__main__":

    # Seed
    np.random.seed(123)

    # Params
    length, paths = 20, 10

    # Data
    X_train = np.random.normal(size=(length, paths))
    y_train = np.random.normal(size=(length, paths))
    y_test = np.random.normal(size=(length, paths))

    # from GridSearchCV
    # -----------------

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor()
    )
    param_grid = {
        'randomforestregressor__max_features': [1.0, "sqrt", "log2"],
        'randomforestregressor__bootstrap': [True, False],
        'randomforestregressor__criterion': ["squared_error", "absolute_error"]
    }
    model = GridSearchCV(pipeline, param_grid=param_grid, cv=10)
    # Predict
    model.fit(X_train, y_train).predict(y_test)


    # from Pipeline with GridSearchCV
    # -------------------------------

    param_grid = {
        'max_features': [1.0, "sqrt", "log2"],
        'bootstrap': [True, False],
        'criterion': ["squared_error", "absolute_error"]
    }
    pipeline = make_pipeline(
        StandardScaler(),
        GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=10)
    )
    # Predict
    pipeline.fit(X_train, y_train).predict(y_test)

    # From TuneEstimator
    # ------------------

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor()
    )
    model = TuneEstimator(pipeline)
    # Check automated distributions
    print(model.get_param_distributions())
    # ...
    # Predict
    model.fit(X_train, y_train).predict(y_test)

    # From Pipeline with TuneEstimator
    # --------------------------------

    pipeline = make_pipeline(
        StandardScaler(),
        TuneEstimator(RandomForestRegressor())
    )
    # Predict
    pipeline.fit(X_train, y_train).predict(y_test)
    