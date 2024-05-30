import numpy as np
from sklearn.ensemble import RandomForestRegressor
from autotuner import TuneEstimator, TuneConfig


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
    
    model = TuneEstimator(estimator, TuneConfig(verbose=True))
    
    estimator.fit(X_train, y_train).predict(y_test)
    # ...
    
    model.fit(X_train, y_train).predict(y_test)
    # ...
    
  