from typing import Any, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import sklearn
from carla.data.catalog.online_catalog import DataCatalog, OnlineCatalog
from carla.data.catalog.own_catalog import DataCatalog, OwnCatalog
from carla.data.load_catalog import load
from carla.models.api import MLModel

from .load_model import load_online_model, load_trained_model, save_model
from .train_model import train_model

class MLModelCatalog(MLModel):
    """
    Use pretrained classifier.

    Parameters
    ----------
    data : data.catalog.DataCatalog Class
        Correct dataset for ML model.
    model_type : {'ann', 'linear', 'forest'}
        The model architecture. Artificial Neural Network, Logistic Regression, and Random Forest respectively.
    backend : {'tensorflow', 'pytorch', 'sklearn', 'xgboost'}
        Specifies the used framework. Tensorflow and PyTorch only support 'ann' and 'linear'. Sklearn and Xgboost only support 'forest'.
    cache : boolean, default: True
        If True, try to load from the local cache first, and save to the cache if a download is required.
    models_home : string, optional
        The directory in which to cache data; see :func:`get_models_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to the read model function
    load_online: bool, default: True
        If true, a pretrained model is loaded. If false, a model is trained.

    Methods
    -------
    predict:
        One-dimensional prediction of ml model for an output interval of [0, 1].
    predict_proba:
        Two-dimensional probability prediction of ml model

    Returns
    -------
    None
    """

    def __init__(
        self,
        data: DataCatalog,
        model_type: str,
        backend: str,
        cache: bool = True,
        models_home: str = None,
        load_online: bool = True,
        **kws,
    ) -> None:
        """
        Constructor for pretrained ML models from the catalog.

        Possible backends are currently "pytorch", "tensorflow" for "ann" and "linear" models.
        Possible backends are currently "sklearn", "xgboost" for "forest" models.

        """
        # **kws is used in the constructor's parameter list to allow passing additional keyword arguments when creating an instance of the MLModelCatalog class. 
        # **kws is a syntax in Python that allows you to capture additional keyword arguments passed to a function or method as a dictionary.
        self._model_type = model_type
        self._backend = backend
        self._categorical = data.categorical
        self.name = data.name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._backend == "pytorch":
            ext = "pt"
        elif self._backend == "tensorflow":
            ext = "h5"
        elif self._backend == "sklearn":
            ext = "skjoblib"
        elif self._backend == "xgboost":
            ext = "xgjoblib"
        else:
            raise ValueError(
                'Backend not available, please choose between "pytorch", "tensorflow", "sklearn", or "xgboost".'
            )
        super().__init__(data)

        # Only datasets that we defined in our own Class named OwnCatalog
        if isinstance(data, OwnCatalog):
            self._feature_input_order = data._column_names

        # Only datasets in our catalog have a saved yaml file   
        elif isinstance(data, OnlineCatalog):
            # Load catalog
            catalog_content = ["ann", "linear"]
            catalog = load("mlmodel_catalog.yaml", data.name, catalog_content)  # type: ignore

            if model_type not in catalog:
                raise ValueError("Model type not in model catalog")

            self._catalog = catalog[model_type][self._backend]
            self._feature_input_order = self._catalog["feature_order"]
        else:
            if data._identity_encoding:
                encoded_features = data.categorical
            else:
                encoded_features = list(
                    data.encoder.get_feature_names(data.categorical)
                )

            self._catalog = None

            if model_type == "forest":
                self._feature_input_order = list(np.sort(data.continuous))
            else:
                self._feature_input_order = list(
                    np.sort(data.continuous + encoded_features)
                )
        
        if load_online:
            self._model = load_online_model(
                model_type, data.name, ext, cache, models_home, **kws
            )

    def _test_accuracy(self):
        # get preprocessed data
        x_test = self.data.df_test
        y_test = self.data.target_test
        pred = self.predict(x_test).cpu()
        print('check probability', pred)
        prediction = [0 if prob < 0.5 else 1 for prob in pred]

        print('set of prediction and test set values', set(prediction), set(y_test))
        print(f"test AUC for model: {sklearn.metrics.roc_auc_score(y_test, pred.detach().numpy())}")
        
        print(f"test accuracy for model: {sklearn.metrics.roc_auc_score(y_test, prediction)}")

    # An @property is used to define getter methods for class attributes. 
    # Getter methods allow you to access the attributes of an object as if they were regular attributes, even though they might involve some computation or data retrieval. 
    @property
    def feature_input_order(self) -> List[str]:
        # This property allows you to access the ordered list of feature names for the ML model. 
        # It returns the _feature_input_order attribute, which is a list of feature names determined based on the dataset being used.
        """
        Saves the required order of feature as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        ordered_features : list of str
            Correct order of input features for ml model
        """
        return self._feature_input_order

    @property
    def model_type(self) -> str:
        # This property allows you to access the type of the ML model, such as an 'ann' or 'linear'
        # It returns the '_model_type' attribute
        """
        Describes the model type

        E.g., ann, linear

        Returns
        -------
        backend : str
            model type
        """
        return self._model_type

    @property
    def backend(self) -> str:
        # This property allows you to access the backend or framework being used for the ML model, such as 'tensorflow', 'pytorch', 'sklearn', or 'xgboost'. 
        # It returns the _backend attribute.
        """
        Describes the type of backend which is used for the ml model.

        E.g., tensorflow, pytorch, sklearn, ...

        Returns
        -------
        backend : str
            Used framework
        """
        return self._backend

    @property
    def raw_model(self) -> Any:
        # This property provides access to the raw ML model built using its framework. 
        # It returns the _model attribute, which stores the loaded or trained ML model.
        """
        Returns the raw ML model built on its framework

        Returns
        -------
        ml_model : tensorflow, pytorch, sklearn model type
            Loaded model
        """
        return self._model

    def predict(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]:
        """
        One-dimensional prediction of ml model for an output interval of [0, 1]

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction for interval [0, 1] with shape N x 1
        """
        if self._backend == "pytorch":
            return self.predict_proba(x)
            raise ValueError(
                'Incorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]:
        """
        Two-dimensional probability prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction with shape N x 2
        """
        if self._backend == "pytorch":
            self._model = self._model.to(self.device) # Keep model and input on the same device
            _x = x.clone()
            tensor_output = torch.is_tensor(x) # If the input was a tensor, return a tensor. Else return a np array.
            _x = _x.long().to(self.device)
            output = self._model(_x, mode='inference') # inference
            if tensor_output:
                return output
            else:
                return output.detach().cpu().numpy()

        else:
            raise ValueError(
                'Incorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    @property
    def tree_iterator(self):
        # This property is specific to tree-based models like random forests. 
        # It returns a list of individual trees that make up the forest. 
        # The exact behavior depends on the model type and backend.
        """
        A method needed specifically for tree methods. This method should return a list of individual trees that make up the forest.

        Returns
        -------

        """
        if self.model_type != "forest":
            return None
        elif self.backend == "sklearn":
            return self._model
        elif self.backend == "xgboost":
            # make a copy of the trees, else feature names are not saved
            booster_it = [booster for booster in self.raw_model.get_booster()]
            # set the feature names
            for booster in booster_it:
                booster.feature_names = self.feature_input_order
            return booster_it

    def train(
        self,
        dataset_name: str,
        dataset_manager,
        optimizer_name="Nadam",
        learning_rate=None,
        weight_decay=None,
        epochs=None,
        batch_size=None,
        force_train=False,
        hidden_sizes=None,
        dropout=0.1,
        n_estimators=5,
        max_depth=5,
        vocab_size=None
    ):
        """

        Parameters
        ----------
        learning_rate: float
            Learning rate for the training.
        epochs: int
            Number of epochs to train for.
        batch_size: int
            Number of samples in each batch
        force_train: bool
            Force training, even if model already exists in cache.
        hidden_sizes: list[int]
            hidden_sizes[i] contains the number of nodes in layer [i]
            hidden_sizes[-1] contains the number of lstm layers
        n_estimators: int
            Number of estimators in forest.
        max_depth: int
            Max depth of trees in the forest.

        Returns
        -------

        """
        save_string_list = [str(dataset_name), str(optimizer_name), str(learning_rate), str(weight_decay), str(epochs), str(batch_size), str(hidden_sizes[0]), str(hidden_sizes[1]), str(hidden_sizes[2]), str(dropout)]
        if self.model_type == "linear" or self.model_type == "forest":
            save_name = f"{self.model_type}"
        elif self.model_type == "ann":
            save_name = f"{self.model_type}_layers_{save_string_list}"
        elif self.model_type == "lstm":
            save_name = f"{self.model_type}_layers_{save_string_list}"
        else:
            raise NotImplementedError("Model type not supported:", self.model_type)

        # try to load the model from disk, if that fails train the model instead.
        self._model = None
        if not force_train:
            self._model = load_trained_model(
                save_name=save_name, data_name=self.name, backend=self.backend
            )

            # sanity check to see if loaded model accuracy makes sense
            if self._model is not None:
                self._test_accuracy()

        # if model loading failed or force_train flag set to true.
        if self._model is None or force_train:
            # get preprocessed data
            x_train = self.data.df_train
            x_test = self.data.df_test
            y_train = self.data.target_train
            y_test = self.data.target_test

            self._model = train_model(
                self,
                dataset_manager,
                x_train,
                y_train,
                x_test,
                y_test,
                optimizer_name,
                learning_rate,
                weight_decay,
                epochs,
                batch_size,
                hidden_sizes,
                dropout,
                n_estimators,
                max_depth,
                vocab_size
            )

            save_model(
                model=self._model,
                save_name=save_name,
                data_name=self.data.name,
                backend=self.backend,
            )
