from typing import List

import numpy as np
import scipy
from scipy.spatial import distance
from sklearn.metrics import pairwise
import pandas as pd
import torch
from carla.evaluation.api import Evaluation
import torch.nn.functional as F
import scipy.stats

class Distance(Evaluation):
    """
    Calculates the L0, L1, L2, and L-infty distance measures.
    """

    def __init__(self, dataset_manager, mlmodel):
        super().__init__(dataset_manager, mlmodel)
        self.dataset_manager = dataset_manager
        self.columns = ["len CF", "L1", "L2", "Linf", 'EMD', "DL Edit",'# changes', 'LSP', 'diversity']

    def get_evaluation(self, factuals, counterfactuals):
        distances = self._get_distances(factuals, counterfactuals)

        return pd.DataFrame(distances, columns=self.columns)

    # Jaccard Distance
    def jaccard_distance(self, set1, set2):
        intersection = np.sum(np.logical_and(set1, set2))
        union = np.sum(np.logical_or(set1, set2))
        return 1 - (intersection / union)

    # Earth Mover's Distance (EMD)
    def earth_mover_distance(self, p, q):
        return scipy.stats.wasserstein_distance(p, q)
    
    def _get_delta(self, factual: torch.tensor, counterfactual: torch.tensor) -> torch.tensor:
        """
        Compute difference between original factual and counterfactual

        Parameters
        ----------
        factual: np.ndarray
            Normalized and encoded array with factual data.
            Shape: NxM
        counterfactual: : np.ndarray
            Normalized and encoded array with counterfactual data.
            Shape: NxM

        Returns
        -------
        np.ndarray
        """
        are_equal = np.array_equal(counterfactual, factual)
        if are_equal==True:
            print('mistake')

        return counterfactual - factual
    
    def diversity_loss(self, matrix):
        emd_sum = 0
        num_samples = matrix.shape[0]
        matrix = np.argmax(matrix, axis=2)

        if num_samples>1:
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    emd = self.earth_mover_distance(matrix[i], matrix[j])
                    emd_sum += emd

            average_emd = emd_sum / (num_samples * (num_samples - 1) / 2)
        else:
            average_emd = 0
        return average_emd
    

    def _get_distances(self, 
        factual: torch.tensor, counterfactual: torch.tensor
    ) -> List[List[float]]:
        """
        Computes distances.
        All features have to be in the same order (without target label).

        Parameters
        ----------
        factual: np.ndarray
            Normalized and encoded array with factual data.
            Shape: NxM
        counterfactual: np.ndarray
            Normalized and encoded array with counterfactual data
            Shape: NxM

        Returns
        -------
        list: distances 1 to 4
        """

        # Replicate factual x times along the first axis (to match the shape of the second array)
        if factual.shape != counterfactual.shape:
            factual = np.tile(factual, (counterfactual.shape[0], 1, 1))

        if factual.shape != counterfactual.shape: #check if it is still the case
            raise ValueError("Shapes of factual and counterfactual have to be the same")

        # get difference between original and counterfactual
        # Calculate the distances
        d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, = [], [], [], [], [], [], [], [], [], [], []
        delta = self._get_delta(factual, counterfactual)
        for i in range(0,len(delta)):
            d0.append(np.round(np.count_nonzero(np.argmax(counterfactual[i], axis=1)),2))
            d1.append(np.round(np.sum(np.abs(delta[i])),2))
            d2.append(np.round(np.sqrt(np.sum((delta[i]) ** 2)),2))
            d3.append(np.round(np.max(np.abs(delta[i])),2))
            d4.append(self.dataset_manager.edit_distance(factual[i], counterfactual[i], False))
            delta_argmax = np.argmax(factual[i], axis=1) != np.argmax(counterfactual[i], axis=1)
            d8.append(np.count_nonzero(delta_argmax, axis=0))
            d9.append(np.argmax(np.argmax(factual[i], axis=1) != np.argmax(counterfactual[i], axis=1)))
            d10.append(np.round(self.diversity_loss(counterfactual),2))
        d5.append(pairwise.pairwise_distances(np.argmax(factual, axis=2), np.argmax(counterfactual, axis=2), metric=self.jaccard_distance)[0,:].tolist())
        d6.append((1 - pairwise.cosine_distances(np.argmax(factual, axis=2), np.argmax(counterfactual, axis=2)))[0,:].tolist())
        d7.append((np.array([[self.earth_mover_distance(p, q) for q in np.argmax(counterfactual, axis=2)] for p in np.argmax(factual, axis=2)])[0,:]).tolist())
        d5 = np.round(d5[0],2)
        d6 = np.round(d6[0],2)
        d7 = np.round(d7[0],2)

        return [[d0[i], d1[i], d2[i], d3[i], d7[i], d4[i], d8[i], d9[i], d10[i]] for i in range(len(d1))]
