<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<p align="center"><img src="docs/logo.png" alt="logo" width="90%" height="90%"></p>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
        <ul>
            <li><a href="#introduction">Introduction</a></li>
        </ul>
        <ul>
            <li><a href="#why-optuna">Why Optuna?</a></li>
        </ul>
        <ul>
            <li><a href="#built-with">Built With</a></li>
        </ul>
    </li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#reference">Reference</a></li>
    
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

### Introduction

`autotuner` is an automated hyper-parameter tuning for `scikit-learn` estimators using `optuna`:

> `optuna` is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.

* **Eager search spaces**: Automated search for optimal hyperparameters using Python conditionals, loops, and syntax
* **State-of-the-art algorithms**: Efficiently search large spaces and prune unpromising trials for faster results
* **Easy parallelization**: Parallelize hyperparameter searches over multiple threads or processes without modifying code

Optuna has integration features with various third-party libraries. Integrations can be found in [`optuna/optuna-integration`](https://optuna.readthedocs.io/en/stable/reference/integration.html). 

`Autotune` is designed to automatically suggest hyperparameters that are fed into a trial object. It works with both `scikit-learn` estimators and pipeline.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Why Optuna?

Optuna enables efficient hyperparameter optimization by adopting state-of-the-art algorithms for sampling hyperparameters and pruning efficiently unpromising trials.

* **Sampling Algorithms**: *Samplers basically continually narrow down the search space using the records of suggested parameter values and evaluated objective values, leading to an optimal search space which giving off parameters leading to better objective values. `optuna` provides the following sampling algorithms:*
  * *Grid Search*
  * *Random Search*
  * *Tree-structured Parzen Estimator algorithm*
  * *CMA-ES based algorithm*
  * *Gaussian process-based algorithm*
  * *Algorithm to enable partial fixed parameters*
  * *Nondominated Sorting Genetic Algorithm II*
  * *A Quasi Monte Carlo sampling algorithm*
  
* **Pruning Algorithms**: *Pruners automatically stop unpromising trials at the early stages of the training (a.k.a., automated early-stopping). Optuna provides the following pruning algorithms:*
  * *Median pruning algorithm*
  * *Non-pruning algorithm*
  * *Algorithm to operate pruner with tolerance*
  * *Algorithm to prune specified percentile of trials*
  * *Asynchronous Successive Halving algorithm*
  * *Hyperband algorithm*
  * *Threshold pruning algorithm*
  * *A pruning algorithm based on Wilcoxon signed-rank test*

More information could be found in the [Official Documentation](https://optuna.readthedocs.io/en/stable/tutorial/index.html).

### Built With

* `optuna = "^3.6.1"`
* `optuna-integration = "^3.6.0"`
* `numpy = "^1.26.4"`
* `scikit-learn = "^1.5.0"`
* `pandas = "^2.2.2"`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Installation

To get started with `autotuner`, you can clone the repository to your local machine. Ensure you have Git installed, then run the following command:

```sh
$ git clone https://github.com/ActurialCapital/autotuner.git
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Getting Started

Once you have cloned the repository, you can start using `autotuner` to optimize `scikit-learn` hyperparameters.

```python
>>> from sklearn.ensemble import RandomForestRegressor
>>> from autotuner import TuneEstimator
>>> model = TuneEstimator(RandomForestRegressor())
```

Get hyperparameters distributions:

```python
>>> model.get_param_distributions()
# {'n_estimators': IntDistribution(high=300, log=False, low=10, step=1),
#  'max_depth': IntDistribution(high=11, log=False, low=1, step=1),
#  'min_impurity_decrease': FloatDistribution(high=0.5, log=True, low=1e-09, step=None),
#  'max_features': FloatDistribution(high=1.0, log=False, low=0.4, step=None),
#  'min_samples_split': IntDistribution(high=10, log=False, low=2, step=1),
#  'min_samples_leaf': IntDistribution(high=6, log=False, low=2, step=1),
#  'bootstrap': CategoricalDistribution(choices=(True, False)),
#  'criterion': CategoricalDistribution(choices=('squared_error', 'absolute_error'))}
```

Fit the estimator:

```python
>>> model.fit(X_train, y_train)
```

Extract best parameters:

```python
>>> model.best_params_
# {'n_estimators': 237,
#  'max_depth': 3,
#  'min_impurity_decrease': 2.9704981079087477e-09,
#  'max_features': 0.47361313286263895,
#  'min_samples_split': 3,
#  'min_samples_leaf': 6,
#  'bootstrap': True,
#  'criterion': 'squared_error'}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the BSD-3 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Reference -->
## Reference

* Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
* Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.
