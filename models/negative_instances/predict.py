from typing import Any

import numpy as np
import pandas as pd


from typing import List

import torch
import numpy as np

#we need the cutoff to determine which instances are negative
from sklearn.metrics import roc_curve

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def predict_negative_instances(model: Any, dataset: np.ndarray, threshold: int, data_type: str, target: list) -> np.ndarray:
    """Predicts the data target and retrieves the negative instances. (H^-)

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    model : PyTorch Model
        Model object used for predictions
    data : np.ndarray
        3D NumPy array representing the dataset used for predictions (sequences, prefix_length, vocab_size)
        
    Returns
    -------
    negative_data : np.ndarray
        3D NumPy array containing negative predicted instances
    """
    negative_data = []
    if data_type == 'train':
        data = dataset.df_train
        label = dataset.train_label
    elif data_type == 'test':
        data = dataset.df_test
        label = dataset.test_label

    if target[0]==0:
        # Retrieve the indices where the label is zero
        one_indices = [index for index, value in enumerate(label) if value == 1]
        # Use zero_indices to select the tensors where the label is zero
        data = data[one_indices,:,:]

    elif target[0]==1:
        # Retrieve the indices where the label is zero
        zero_indices = [index for index, value in enumerate(label) if value == 0]
        # Use zero_indices to select the tensors where the label is zero
        data = data[zero_indices,:,:]

    
  
    for i in range(0,data.shape[0]):
        sequence = data[i,:,:]
        sequence_tensor = sequence.unsqueeze(0)  # Add batch dimension
        predictions = predict_label(model, sequence_tensor, as_prob=True)[0] # Predict the label for the sequence 

        if target[0]==0:
            if predictions.item() > threshold: # Check if the predicted label is negative
                negative_data.append(sequence)
        
        elif target[0]==1:
            if predictions.item() < threshold: # Check if the predicted label is negative
                negative_data.append(sequence)
    negative_data = torch.stack(negative_data, dim=0)
    return negative_data

def predict_label(model: Any, df: np.ndarray, as_prob: bool = False) -> np.ndarray:
    """Predicts the data target

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    df : np.ndarray
        3D NumPy array used for predictions
    Returns
    -------
    predictions :  1D NumPy array with predictions
    """

    if model.backend =='pytorch':
        predictions = model.predict(df)

    if not as_prob:
        predictions = predictions.round()
    return predictions
