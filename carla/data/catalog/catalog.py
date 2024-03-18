from abc import ABC
from typing import Callable, List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator

from carla.data.pipelining import (
    decode,
    descale,
    encode,
    fit_encoder,
    fit_scaler,
    scale,
)

from ..api import Data


class DataCatalog(Data, ABC):
    # The DataCatalog class provides a framework for handling datasets with specific preprocessing requirements. 
    # It allows for scaling, encoding, and decoding data while keeping track of different data splits (train and test) and the dataset's name. 
    # Subclasses like OnlineCatalog and CsvCatalog can use this base class to manage datasets with specific loading mechanisms.
    """
    Generic framework for datasets, using sklearn processing. This class is implemented by OnlineCatalog and CsvCatalog.
    OnlineCatalog allows the user to easily load online datasets, while CsvCatalog allows easy use of local datasets.

    Parameters
    ----------
    data_name: str
        What name the dataset should have.
    df: pd.DataFrame
        The complete Dataframe. This is equivalent to the combination of df_train and df_test, although not shuffled.
    df_train: pd.DataFrame
        Training portion of the complete Dataframe.
    df_test: pd.DataFrame
        Testing portion of the complete Dataframe.
    scaling_method: str, default: MinMax
        Type of used sklearn scaler. Can be set with the property setter to any sklearn scaler.
        Set to "Identity" for no scaling.
    encoding_method: str, default: OneHot_drop_binary
        Type of OneHotEncoding {OneHot, OneHot_drop_binary}. Additional drop binary decides if one column
        is dropped for binary features. Can be set with the property setter to any sklearn encoder.
        Set to "Identity" for no encoding.

    Returns
    -------
    Data
    """

    def __init__(
        self,
        data_name: str,
        df_train,
        df_test,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot_drop_binary",
    ):
        self._df_train = df_train
        self._df_test = df_test
        self._name = data_name

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test.copy()
    
    @property
    def name(self) -> str:
        return self._name

    def get_pipeline_element(self, key: str) -> Callable:
        """
        Returns a specific element of the transformation pipeline.

        Parameters
        ----------
        key : str
            Element of the pipeline we want to return

        Returns
        -------
        Pipeline element
        """
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("scaler", lambda x: scale(self.scaler, self.continuous, x)),
            ("encoder", lambda x: encode(self.encoder, self.categorical, x)),
        ]

    def __init_inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("encoder", lambda x: decode(self.encoder, self.categorical, x)),
            ("scaler", lambda x: descale(self.scaler, self.continuous, x)),
        ]
