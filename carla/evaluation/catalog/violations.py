from typing import List

import numpy as np
import pandas as pd
import torch
from carla.data.api import Data
from carla.evaluation.api import Evaluation


class ConstraintViolation(Evaluation):
    """
    Computes the constraint violation per factual as dataframe
    """

    def __init__(self, dataset_manager, mlmodel, constraintminer):
        super().__init__(dataset_manager, mlmodel, constraintminer)
        self.constraint_miner = constraintminer
        self.mlmodel = mlmodel
        self.columns = ["Violations (%)"]

    def get_evaluation(self, factuals, counterfactuals):
        violations = self.constraint_violation(counterfactuals)

        return pd.DataFrame(violations, columns=self.columns)

    def _intersection(list1: List, list2: List):
        """Compute the intersection between two lists"""
        return list(set(list1) & set(list2))


    def constraint_violation(self, counterfactuals: torch.tensor
    ) -> List[List[float]]:
        """
        Counts constraint violation per counterfactual

        Parameters
        ----------
        data:

        counterfactuals:
            Normalized and encoded counterfactual examples.
        factuals:
            Normalized and encoded factuals.

        Returns
        -------

        """
        violation_list = []
        for i in range(0, counterfactuals.shape[0]):
            violations, constraints = self.constraint_miner.count_violations(torch.tensor(counterfactuals[i], dtype=torch.float32).squeeze(), constraint_list = 'total')
            violation_perc = violations/len(constraints)*100
            violation_list.append(violation_perc)
        return violation_list
