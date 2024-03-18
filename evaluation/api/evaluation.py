from abc import ABC, abstractmethod

import torch


class Evaluation(ABC):
    def __init__(self, dataset_manager, mlmodel= None, hyperparameters: dict = None):
        """

        Parameters
        ----------
        mlmodel:
            Classification model. (optional)
        hyperparameters:
            Dictionary with hyperparameters, could be used to pass other things. (optional)
        """
        self.dataset_manager = dataset_manager
        self.mlmodel = mlmodel
        self.hyperparameters = hyperparameters

    @abstractmethod
    def get_evaluation(
        self, factuals: torch.tensor, counterfactuals: torch.tensor
    ) -> torch.tensor:
        """Compute evaluation measure"""
