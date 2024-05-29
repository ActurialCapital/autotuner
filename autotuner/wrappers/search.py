from optuna import samplers


class WrapSearch:
    """
    A class for creating an Optuna sampler based on a specified search 
    algorithm.

    This class facilitates the creation of different Optuna samplers like TPE 
    (Tree-structured Parzen Estimator) or Random based on the given search 
    algorithm identifier. It also incorporates a seed for random state 
    initialization, ensuring reproducibility in the sampling process.

    Parameters
    ----------
    search_algorithm : str
        The name of the search algorithm to use for the sampler. User can
        also pass the optuna sampler directly.
    seed : int, optional
        Seed for random number generator. Defaults to None.

    Methods
    -------
    create_sampler()
        Creates and returns an Optuna sampler based on the specified search 
        algorithm.

    """

    __slots__ = ["search_algorithm", "seed"]

    def __init__(
        self,
        search_algorithm: str | samplers.BaseSampler, 
        seed: int = None
    ):
        self.search_algorithm = search_algorithm
        self.seed = seed

    def create_sampler(self) -> samplers.BaseSampler:
        """
        Creates an Optuna sampler based on the provided search algorithm 
        identifier.

        This method selects the appropriate sampler from a predefined set based 
        on the provided string identifier. If no valid identifier is provided, 
        it raises a ValueError.

        Returns
        -------
        opt.samplers.BaseSampler
            The Optuna sampler corresponding to the search algorithm.

        Raises
        ------
        ValueError
            If the specified search algorithm name is not recognized.

        Examples
        --------
        Using optuna sampler

        !!! note

            Passing an optuna sampler in this context is just for 
            proof-of-concept.

        ```pycon
        >>> from optuna import samplers
        >>> search = WrapSearch(samplers.TPESampler())
        >>> search.create_sampler()
        <optuna.samplers._tpe.sampler.TPESampler at 0x138d08150>
        ```

        Using key
        ```pycon
        >>> search = WrapSearch("tpe")
        >>> search.create_sampler()
        <optuna.samplers._tpe.sampler.TPESampler at 0x138d08150>
        ```
        
        ```pycon
        >>> search = WrapSearch("random")
        >>> search.create_sampler()
        <optuna.samplers._random.RandomSampler at 0x1389fef50>
        ```

        """
        if isinstance(self.search_algorithm, samplers.BaseSampler):
            return self.search_algorithm

        repository = {
            "tpe": samplers.TPESampler(
                seed=self.seed,
                multivariate=True,
                constant_liar=True
            ),
            "random": samplers.RandomSampler(seed=self.seed),
        }

        if self.search_algorithm not in repository:
            raise ValueError(
                f"'{self.search_algorithm}' is not a valid search algorithm. "
                f"Available algorithms are {list(samplers.keys())}."
            )

        return repository[self.search_algorithm]
