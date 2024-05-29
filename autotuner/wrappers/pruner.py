from optuna import pruners


class WrapPrune:
    """
    A class for creating an optuna pruner based on a specified identifier, 
    such as: 

    * Successive Halving
    * Hyperband
    * Median Pruner

    based on a string identifier. 

    It can also return a 'no operation' pruner if no pruning is desired.

    Parameters
    ----------
    pruner : str | pruners.BasePruner
        The name of the pruner to create. Valid pruners are:

        * `asha`
        * `hyperband`,
        * `median`
        * None 
        * False

        If set to None or False, `NopPruner()` will be applied. User can
        also pass the optuna pruner directly.

    Methods
    -------
    create_pruner()
        Creates and returns an Optuna pruner based on the specified identifier.

    """

    def __init__(self, pruner: str):
        self.pruner = pruner

    def create_pruner(self) -> pruners.BasePruner:
        """
        Creates an Optuna pruner based on the provided identifier.

        This method selects the appropriate pruner from a predefined set based 
        on the provided string identifier. If no valid identifier is provided, 
        it raises a ValueError.

        Returns
        -------
        pruners.BasePruner
            The Optuna pruner corresponding to the identifier.

        Raises
        ------
        ValueError
            If the specified pruner name is not recognized.

        Examples
        --------
        Using optuna pruner

        !!! note

            Passing an optuna sampler in this context is just for 
            proof-of-concept.

        ```pycon
        >>> from optuna import pruners
        >>> prune = WrapPrune(pruners.HyperbandPruner())
        >>> prune.create_pruner()
        <optuna.pruners._hyperband.HyperbandPruner at 0x138a914d0>
        ```

        Using key
        ```pycon
        >>> prune = WrapPrune("hyperband")
        >>> prune.create_pruner()
        <optuna.pruners._hyperband.HyperbandPruner at 0x138e36290>
        ```
        """
        if isinstance(self.pruner, pruners.BasePruner):
            return self.pruner

        repository = {
            "asha": pruners.SuccessiveHalvingPruner(),
            "hyperband": pruners.HyperbandPruner(),
            "median": pruners.MedianPruner(),
            False: pruners.NopPruner(),
            None: pruners.NopPruner(),
        }

        if self.pruner not in repository:
            raise ValueError(
                f"'{self.pruner}' is not a valid pruner. "
                f"Valid pruners are {list(pruners.keys())}."
            )

        return repository[self.pruner]
