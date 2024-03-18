import timeit
from typing import List
import torch
import numpy as np
import pandas as pd

from carla.evaluation.api import Evaluation
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod


class Benchmark:
    """
    The benchmarking class contains all measurements.
    It is possible to run only individual evaluation metrics or all via one single call.

    For every given factual, the benchmark object will generate one counterfactual example with
    the given recourse method.

    Parameters
    ----------
    mlmodel: carla.models.MLModel
        Black Box model we want to explain.
    recourse_method: carla.recourse_methods.RecourseMethod
        Recourse method we want to benchmark.
    factuals: pd.DataFrame
        Instances for which we want to find counterfactuals.
    """

    def __init__(
        self,
        dataset_manager,
        mlmodel: MLModel,
        
        recourse_method: RecourseMethod,
        factuals: torch.tensor,
        threshold
    ) -> None:
        self.dataset_manager = dataset_manager
        self.mlmodel = mlmodel
        self._recourse_method = recourse_method
        self._factuals = np.array(factuals.squeeze())
        start = timeit.default_timer()
        print('generating the counterfactuals:')
        self._counterfactuals = np.array(torch.tensor(recourse_method.get_counterfactuals(factuals)))
        stop = timeit.default_timer()
        self.timer = stop - start

    def run_benchmark(self, measures: List[Evaluation]) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Parameters
        ----------
        measures : List[Evaluation]
            List of Evaluation measures that will be computed.

        Returns
        -------
        pd.DataFrame
        """
        pipeline = [
            measure.get_evaluation(
                counterfactuals=self._counterfactuals, factuals=self._factuals
            )
            for measure in measures
        ]
        factual_list = np.argmax(self._factuals, axis=1).tolist()
        argmax_cfs = np.argmax(self._counterfactuals, axis=2)
        cf_list = [row.tolist() for row in argmax_cfs]
        # Create DataFrames from the lists of lists with column names
        # Create DataFrames from the lists of lists
        df1 = {'factual': [factual_list]*self._counterfactuals.shape[0], 'counterfactual': cf_list}
        df1 = pd.DataFrame(df1)
        # Concatenate the DataFrames along the columns
        output = pd.concat(pipeline, axis=1)
        result_df = pd.concat([df1, output], axis=1)
        
        return result_df
