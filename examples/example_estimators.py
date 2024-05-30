import numpy as np
from sklearn.ensemble import RandomForestRegressor
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


    # RandomForestRegressor
    
    
    estimator = RandomForestRegressor()
    
    tune = TuneEstimator(estimator)
    
    estimator.fit(X_train, y_train).predict(y_test)
    # ...
    
    tune.fit(X_train, y_train).predict(y_test)
    # ...
    
  