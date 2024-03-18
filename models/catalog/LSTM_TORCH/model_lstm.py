import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import logging
logging.getLogger().setLevel(logging.INFO)

"""
The training parameters for the Long Short-Term Memory (LSTM) neural network
The first two values of the hidden_sizes are the hidden dimensions, the last value is the lstm size. 
The hyperparameters used in Camargo et al.:
    - dropout : 0.2
    - batch size: 32  # Usually 32/64/128/256
    - epochs: 200
    - hidden_sizes: 50/100
    - optimizer Adam (lr=0.001), SGD (lr=0.01), Adagrad (lr=0.01)
"""

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, dropout, hidden_sizes):
        super().__init__()
        print('vocab_size', vocab_size, 'dropout', dropout, 'hidden sizes', hidden_sizes)
        self.hidden_dim1 = hidden_sizes[0]
        self.hidden_dim2 = hidden_sizes[1]
        self.vocab_size = vocab_size
        self.lstm_dropout = dropout
        self.lstm_size= hidden_sizes[2]
        self.lstm1 = nn.LSTM(input_size = self.vocab_size, hidden_size = self.hidden_dim1, dropout = self.lstm_dropout, num_layers=self.lstm_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2*self.hidden_dim1, self.hidden_dim2)
        self.fc2= nn.Linear(self.hidden_dim2, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        self.to(self.device)

    def forward(self, x_act, mode):
        # x_act = x_act.to(self.device) # when running on a GPU
        x_act = x_act.float()

        if mode == 'train' or mode=='val':
            sequence_length = (x_act[:, :, 0] != 1).sum(dim=1).cpu()[0].item() # Find the indices with the first occurrence of 1 in the first column (i.e. the sequence length)
            lstm_out1, _ = self.lstm1(x_act)
            final_hidden_state = lstm_out1[:, sequence_length-1, :] # take the last hidden state of a non-padding token

        elif mode=='inference':
            lstm_out1, _ = self.lstm1(x_act)
            final_hidden_state = lstm_out1[:, -1, :] # take the last hidden state

        out = self.fc1(final_hidden_state)
        output = torch.sigmoid(self.fc2(out))
        return output

    """
    def predict(self, data):
        
        predict method for CFE-Models which need this method.

        Parameters
        ----------
        data: Union(torch, list)

        Returns
        -------
        np.array with prediction
        
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
            # input = torch.squeeze(input)
        else:
            print('data', data)
            input = torch.squeeze(data)
            print('input', input)
            breakpoint()
        return self.forward(input).detach().numpy()
    """


class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf

    def __call__(self, model, epoch, metric_val, learning_rate, dropout, hidden_dim1, hidden_dim2, lstm_size):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + '_' + str(learning_rate) + '_' + str(hidden_dim1) + '_' + str(hidden_dim2) + '_' + str(lstm_size) + '_' + str(dropout) + f'_epoch{epoch}.pt')
        save = metric_val < self.best_metric_val if self.decreasing else metric_val > self.best_metric_val
        if save:
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]

