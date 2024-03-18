import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from carla.evaluation.api import Evaluation
import torch

class YNN(Evaluation):
    """
    Computes y-Nearest-Neighbours for generated counterfactuals

    Notes
    -----
    - Hyperparams

        * "y": int
            Number of neighbours to use.
        * "cf_label": int
            What class to use as a target.
    """

    def __init__(self, dataset_manager, mlmodel, hyperparameters):
        super().__init__(dataset_manager, mlmodel, hyperparameters)
        self.dataset_manager = dataset_manager
        self.y = self.hyperparameters["y"]
        self.cf_label = self.hyperparameters["cf_label"]
        self.nn = self.hyperparameters["NN"]
        self.columns = ["y-NN"]
        self.mlmodel

    
    def find_closest_sequences(self, seq1, seq2, num):
        # Calculate the edit distances for each sequence in seq2
        distances = []
        if seq1.shape != seq2.shape:
            seq1 = np.tile(seq1, (seq2.shape[0], 1, 1))
        for i in range(0,seq1.shape[0]):
            distances.append(self.dataset_manager.edit_distance(seq1[i], seq2[i], False))
        # Find the two smallest distances and their corresponding sequences
        # Get the indices that would sort the distances in ascending order
        sorted_indices = np.argsort(distances)
        # Sort the array based on the sorted indices
        sorted_arr = seq2[sorted_indices]
        closest_sequences = sorted_arr[0:num, :,:]

        return closest_sequences
    
    def predict_label(self, neighbours):
        predicted_labels = []
        for i in range(0,neighbours.shape[0]):
            sequence = neighbours[i,:,:]
            sequence_tensor = torch.tensor(sequence).unsqueeze(0)  # Add batch dimension
            prediction = self.mlmodel.predict(sequence_tensor)
            prediction = prediction.round()[0].item()
            predicted_labels.append(prediction)

        return predicted_labels

    def _ynn(self, counterfactuals, num):
        count_ynns = []
        factuals = np.array(self.mlmodel.data.df_train)
        for i in range(0,len(counterfactuals)):
            counterfactual = counterfactuals[i]
            counterfactual = counterfactual[np.newaxis,:,:]
            neighbours = self.find_closest_sequences(counterfactual, factuals, num)
            neighbour_labels = self.predict_label(neighbours)
            count = neighbour_labels.count(self.cf_label)
            count = count/len(neighbour_labels)
            count_ynns.append(count)
            """
            nbrs = NearestNeighbors(n_neighbors=self.y).fit(factuals)
            for i, row in counterfactuals.iterrows():
                if np.any(row.isna()):
                    raise ValueError(f"row {i} did not contain a valid counterfactual")

                knn = nbrs.kneighbors(
                    row.values.reshape((1, -1)), self.y, return_distance=False
                )[0]
                for idx in knn:
                    neighbour = factuals.iloc[idx]
                    neighbour = neighbour.values.reshape((1, -1))
                    neighbour_label = np.argmax(self.mlmodel.predict_proba(neighbour))
                    number_of_diff_labels += np.abs(self.cf_label - neighbour_label)
            """
        return count_ynns
        #return [[d1[i], d2[i], d3[i], d4[i]] for i in range(len(d1))]

    def get_evaluation(self, factuals, counterfactuals):

        ynn = self._ynn(counterfactuals, self.nn)
        return pd.DataFrame(ynn, columns=self.columns)
