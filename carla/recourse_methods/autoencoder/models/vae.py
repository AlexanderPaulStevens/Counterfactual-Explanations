import os
from typing import List, Union
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from carla import log
from carla.recourse_methods.autoencoder.save_load import get_home

#hyperoptimization
import wandb
import logging
logging.getLogger().setLevel(logging.INFO)


tf.compat.v1.disable_eager_execution()

"""
The parameters used for the encoder-decoder of Taymouri et al.
    - hidden size: 200
    - num_layers= 5
    - num_directions:1
    - dropout=0.3
    - epochs: 500
    - lr: 5e-5
"""

class VariationalAutoencoder(nn.Module):
    def __init__(self, data_name: str, layers: int, vocab_size: int, max_prefix_length: int, epochs: int, constraint_violation=None, joint_constraint_in_loss=False):
        """
        Parameters
        ----------
        data_name:
            Name of the dataset, used for the name when saving and loading the model.
        layers:
            layers[0] contains the hidden size
            layers[1] contains the latent size
            layers[2] contains the number of lstm layers
        vocab_size:
            The vocabulary size
        max_prefix_length:
            The maximum length of the prefix sequences
        """
        super(VariationalAutoencoder, self).__init__()
        self.hidden_dim = layers[0]
        self.latent_size = layers[1]
        self.lstm_size = layers[2]
        self.vocab_size = vocab_size
        self._max_prefix_length = max_prefix_length
        self._data_name = data_name
        self.epochs = epochs
        # The VAE components
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=self.vocab_size,
            hidden_size= self.hidden_dim,  # Adjust this as needed
            num_layers=self.lstm_size,  # You can increase layers if needed
            batch_first=True,
            bidirectional=False  # Set to True for bidirectional LSTM
        )

        # the ReLu layer 
        # Latent variable layers
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_size)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_size)
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_dim,  # Should match encoder hidden size
            num_layers=self.lstm_size,  # You can increase layers if needed
            batch_first=True,
            bidirectional=False  # Set to True for bidirectional LSTM
        )
        #self.log_softmax = torch.nn.LogSoftmax(dim=2)
        # Output layers
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        # read the file to load in the joint constraints
        self.joint_constraint_in_loss = joint_constraint_in_loss
        self.constraint_violation = constraint_violation


    def encode(self, input_x):
        input_x = input_x.to(self.device)
        input_x = input_x.float()  # [batch, sequence length, vocab]
        encoder_output, _ = self.encoder(input_x) # pass through lstm  
        mu = self.fc_mu(encoder_output) #previously encoder_output[:,-1,:]
        log_var = self.fc_var(encoder_output) # mu, logvar [batch, sequence length, latent_size]
        return mu, log_var

    def decode(self, z):
        z = z.to(self.device)
        decoder_output, _ = self.decoder(z) # z [batch, sequence length, latent_size]
        output = self.fc_out(decoder_output)
        #output = self.log_softmax(output)
        return output

    def _reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var).to(self.device)
        epsilon = torch.randn_like(std).to(self.device)  # Sample from a Gaussian distribution with mean 0 and std 1
        return mu + std * epsilon

    def forward(self, input_x):
        input_x = input_x.to(self.device)
        mu, log_var = self.encode(input_x)
        z = self._reparametrization_trick(mu, log_var)
        output = self.decode(z)
        return output, mu, log_var

    def predict(self, data):
        return self.forward(data)

    def kld(self, mu, logvar):

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).to(self.device)
        return KLD
    
    def mask_out_tensor(self, tensor):
        # Find the index of the maximum value (EoS token) in each tensor
        tensor_max = tensor.clone().detach().to(self.device)
        _, index = torch.max(tensor_max, dim=2)
        index = index.to(self.device)
        index = index
        result_indexes = []
        for row in index:
            row = row.to(self.device)
            index2 = (row == self.vocab_size-1).nonzero(as_tuple=False).to(self.device)
            if len(index2) > 0:
                result_indexes.append(index2[0, 0].item())
            else:
                result_indexes.append(-1)   # result contains the indexes of where the value 
        for idx in range(tensor.shape[0]):
            if result_indexes[idx] == -1:
                continue
            else:
                for j in range(result_indexes[idx]+1, tensor.shape[1]):
                    tensor[idx][j,:] = torch.tensor([0]*tensor.shape[2])

        return tensor
    
    def fit(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        kl_weight=0.3,
        lambda_reg=1e-6,
        epochs=5,
        lr=1e-3,
        batch_size=128,
        joint_constraint=None
    ):

        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=lambda_reg,
        )

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        log.info("Start training of Variational Autoencoder...")
        for epoch in range(epochs):
            print('epoch', epoch)
            beta = epoch * kl_weight / epochs
            beta2 = 10
            # Initialize the losses
            train_loss = 0
            train_loss_num = 0
            # Train for all the batches
            for data in train_loader:
                data = data.float().to(self.device)      
                padding_tensor = torch.zeros([self.vocab_size]).to(self.device)
                padding_tensor[0] =1
                mask = ~torch.all(torch.eq(data, padding_tensor), dim=2).to(self.device)
                # Apply the mask to ignore padded sequences
                data = data * mask.unsqueeze(-1).float()
                reconstruction, mu, log_var = self(data) # returns the reconstruction, mu and log_var
                reconstruction  = self.mask_out_tensor(reconstruction)
                #reconstruction = reconstruction * mask.unsqueeze(-1).float()
                recon_max  = reconstruction.clone().detach().to(self.device)
                recon_max = torch.argmax(recon_max, dim=2).to(self.device) #argmax the reconstruction
                class_indices = torch.argmax(data, dim=2).to(self.device)
                class_indices = class_indices.view(-1).to(self.device)
                reconstruction_constraint = reconstruction.clone().detach().to(self.device)
                reconstruction = reconstruction.view(-1, self.vocab_size).to(self.device)
                self.recon_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
                recon_loss = self.recon_loss(reconstruction, class_indices).to(self.device)
                assert not torch.isnan(mu).any(), "NaN values detected in mu!"
                assert not torch.isnan(log_var).any(), "NaN values detected in log_var!"
                kld_loss = self.kld(mu, log_var).to(self.device)

                if self.joint_constraint_in_loss:
                    constraint_loss = self.constraint_violation.calculate_joint_constraint_loss(reconstruction_constraint)
                    loss = recon_loss + beta * kld_loss + beta2 * constraint_loss

                else:
                    loss = recon_loss + beta * kld_loss
   
                optimizer.zero_grad() # Update the parameters
                loss.backward() # Compute the loss

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0) #gradient clipping to avoid gradients from exploding

                # Update the parameters
                optimizer.step()

                # Collect the ways
                train_loss += loss.item()
                train_loss_num += 1

            ELBO[epoch] = train_loss / train_loss_num
            if epoch % 10 == 0:
                log.info(
                    "[Epoch: {}/{}] [objective: {:.3f}]".format(
                        epoch, epochs, ELBO[epoch, 0]
                    )
                )

            ELBO_train = ELBO[epoch, 0].round(2)
            log.info("[ELBO train: " + str(ELBO_train) + "]")

        self.save()
        log.info("... finished training of Variational Autoencoder.")

        self.eval()

    def load(self):
        cache_path = get_home()

        if self.joint_constraint_in_loss:
            cache_path = os.path.join(cache_path, 'joint_constraints')
            
        elif self.joint_constraint_in_loss == False:
            cache_path = os.path.join(cache_path, 'no_constraints')

        load_path = os.path.join(
            cache_path,
            "{}_{}_{}_{}_{}.{}".format(self._data_name, self.hidden_dim, self.latent_size, self.lstm_size, self.epochs, "pt"),
        )
        if torch.cuda.is_available() == False:
          self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        else:
          self.load_state_dict(torch.load(load_path))

        self.eval()

        return self

    def save(self):
        cache_path = get_home()

        if self.joint_constraint_in_loss:
            cache_path = os.path.join(cache_path, 'joint_constraints')
            save_path = os.path.join(
                cache_path,
                "{}_{}_{}_{}_{}.{}".format(self._data_name, self.hidden_dim, self.latent_size, self.lstm_size, self.epochs, "pt"),
            )  
        elif self.joint_constraint_in_loss == False:
            cache_path = os.path.join(cache_path, 'no_constraints')

        if not os.path.exists(os.path.join(cache_path)):
            os.makedirs(os.path.join(cache_path))

        save_path = os.path.join(
                cache_path,
                "{}_{}_{}_{}_{}.{}".format(self._data_name, self.hidden_dim, self.latent_size, self.lstm_size, self.epochs, "pt"),
            )
        torch.save(self.state_dict(), save_path)

class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, epoch, metric_val, learning_rate, lstm_size, optimizer):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ +'_'+ str(learning_rate) + '_' + str(lstm_size) + '_' + str(optimizer) +'_' + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'model-ckpt'+'_'+ str(learning_rate) +'_' + str(lstm_size) +'_'+ str(optimizer) +'_' + f'-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
    
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]