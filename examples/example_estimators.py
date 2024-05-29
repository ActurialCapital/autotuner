import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from autotuner import TuneEstimator


if __name__ == "__main__":

    # Seed

    np.random.seed(123)

    # Params

    length = 12
    paths = 2

    # Data

    X_train = np.random.normal(loc=0, scale=0.2, size=(length, paths))
    y_train = np.random.normal(loc=0, scale=0.2, size=(length, paths))
    y_test = np.random.normal(loc=0, scale=0.2, size=(length, paths))

    # Linear regression

    model = TuneEstimator(LinearRegression())
    model.fit(X_train, y_train)
    model.predict(y_test)
    # ...

    # Random forest

    model = TuneEstimator(RandomForestRegressor())
    model.fit(X_train, y_train)
    model.predict(y_test)
    # ...
