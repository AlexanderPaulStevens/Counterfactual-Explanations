from typing import Union

import numpy as np
import torch
import xgboost
import sklearn
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd 
from util.DatasetManager import DatasetManager
from carla.models.catalog.ANN_TF import AnnModel
from carla.models.catalog.ANN_TF import AnnModel as ann_tf
from carla.models.catalog.ANN_TORCH import AnnModel as ann_torch
from carla.models.catalog.LSTM_TORCH import LSTMModel
from carla.models.catalog.LSTM_TORCH import LSTMModel as lstm_torch
from carla.models.catalog.Linear_TF import LinearModel
from carla.models.catalog.Linear_TF import LinearModel as linear_tf
from carla.models.catalog.Linear_TORCH import LinearModel as linear_torch

def train_model(
    catalog_model,
    dataset_manager,
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    hidden_sizes: list,
    dropout: float,
    n_estimators: int,
    max_depth: int,
    vocab_size: int,
) -> Union[LinearModel, AnnModel, LSTMModel, RandomForestClassifier, xgboost.XGBClassifier]:
    """

    Parameters
    ----------
    catalog_model: MLModelCatalog
        API for classifier
    x_train: pd.DataFrame
        training features
    y_train: pd.DataFrame
        training labels
    x_test: pd.DataFrame
        test features
    y_test: pd.DataFrame
        test labels
    learning_rate: float
        Learning rate for the training.
    epochs: int
        Number of epochs to train on.
    batch_size: int
        Size of each batch
    hidden_sizes: list[int]
        hidden_sizes[i] contains the number of nodes in layer [i].
        hidden_sizes[-1] contains the lstm size
    n_estimators: int
        Number of trees in forest
    max_depth: int
        Max depth of trees in forest

    Returns
    -------
    Union[LinearModel, AnnModel, RandomForestClassifier, xgboost.XGBClassifier]
    """
    print(f"balance on test set {y_train.mean()}, balance on test set {y_test.mean()}")
    if catalog_model.backend == "tensorflow":
        if catalog_model.model_type == "linear":
            model = linear_tf(
                dim_input=x_train.shape[1],
                num_of_classes=len(set(y_train)),
                data_name=catalog_model.data.name,
            )  # type: Union[linear_tf, ann_tf]
        elif catalog_model.model_type == "ann":
            model = ann_tf(
                dim_input=x_train.shape[1],
                dim_hidden_layers=hidden_sizes,
                num_of_classes=len(set(y_train)),
                data_name=catalog_model.data.name,
            )
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
        model.build_train_save_model(
            x_train,
            y_train,
            x_test,
            y_test,
            epochs,
            batch_size,
            model_name=catalog_model.model_type,
            vocab_size=vocab_size
        )
        return model.model
    
    elif catalog_model.backend == "pytorch":
        train_dataset = SequenceDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = SequenceDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        if catalog_model.model_type == "linear":
            model = linear_torch(
                dim_input=x_train.shape[1], num_of_classes=len(set(y_train))
            )
        elif catalog_model.model_type == "ann":
            model = ann_torch(
                input_layer=x_train.shape[1],
                hidden_layers=hidden_sizes,
                num_of_classes=len(set(y_train)),
            )
        elif catalog_model.model_type == "lstm":
            model = lstm_torch(
                vocab_size, 
                dropout, 
                hidden_sizes
            )

        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
        _training_torch(
            model,
            train_loader,
            test_loader,
            optimizer_name,
            learning_rate,
            weight_decay,
            epochs
        )
        return model
    elif catalog_model.backend == "sklearn":
        if catalog_model.model_type == "forest":
            random_forest_model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth
            )
            random_forest_model.fit(X=x_train, y=y_train)
            train_score = random_forest_model.score(X=x_train, y=y_train)
            test_score = random_forest_model.score(X=x_test, y=y_test)
            print(
                "model fitted with training score {} and test score {}".format(
                    train_score, test_score
                )
            )
            return random_forest_model
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )

class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X_train = torch.LongTensor(x).to(device)
        self.Y_train = torch.LongTensor(y).to(device)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


def _training_torch(
    model,
    train_loader,
    test_loader,
    optimizer_name,
    learning_rate,
    weight_decay,
    epochs,
):
    loaders = {"train": train_loader, "inference": test_loader}

    # Use GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define the loss
    criterion = nn.BCELoss()

    # declaring optimizer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9, 0.999), amsgrad=False)
    
    # training
    for e in range(epochs):
        print("Epoch {}/{}".format(e, epochs - 1))
        print("-" * 10)
        
        # Each epoch has a training and validation phase
        for phase in ["train", "inference"]:

            running_loss = 0.0
            predictions_all = []
            pred_probs_all = []
            labels_all = []
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            for i, (inputs, labels) in enumerate(loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).type(torch.int64)
                #labels = torch.nn.functional.one_hot(labels, num_classes=2)
                labels = labels.unsqueeze(-1) #you need this if your LSTM model only predicts 1 probability
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    inputs = inputs.long()
                    outputs = model(inputs, mode= phase)
                    loss = criterion(outputs.float(), labels.float())
                    output_detached = outputs.clone().detach().cpu().numpy()
                    pred_probs_all.extend(output_detached)
                    labels_all.extend(labels.cpu())
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                # loss.item() extrats the scalar value of the loss function at a particular step. 
                # inputs.size(0) retrieves the size of the input batch
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_auc = sklearn.metrics.roc_auc_score(labels_all, pred_probs_all)
            print("{} Running loss: {:.4f}  epoch AUC: {:.4f}".format(phase, epoch_loss, epoch_auc))
            
