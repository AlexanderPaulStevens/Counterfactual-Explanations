import copy
import os
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange

from carla import log
from carla.recourse_methods.autoencoder.losses import csvae_loss
from carla.recourse_methods.autoencoder.save_load import get_home

tf.compat.v1.disable_eager_execution()


class CSVAE(nn.Module):
    def __init__(self, data_name: str, layer_list: List[int], vocab_size: int, max_prefix_length: int) -> None:
        super(CSVAE, self).__init__()
        self._sequence_length = max_prefix_length
        # you can also make the hidden dim of the encoder and decoder differrently. For now, we did not do that
        self._hidden_dim = layer_list[0]
        self.z_dim = layer_list[-2]
        self._lstm_size = layer_list[-1]
        self._vocab_size = vocab_size
        self._max_prefix_length = max_prefix_length
        
        #self._input_dim = layers[0]
        #self.z_dim = layers[-1]
        self._data_name = data_name

        # w_dim and labels_dim are fix due to our constraint to binary labeled data
        w_dim = 2
        self._labels_dim = w_dim

        # encoder
        # Replace linear layers of the encoder with LSTM layers
        self.encoder_xy_to_w = nn.LSTM(
            input_size=self._vocab_size + self._labels_dim,  # Input size includes data and labels
            hidden_size=self._hidden_dim,
            num_layers=self._lstm_size,  # You can adjust the number of layers as needed
            batch_first=True,  # Batch dimension is first (batch_size, seq_len, input_size)
        )

        self.mu_xy_to_w = nn.Linear(self._hidden_dim, w_dim)

        self.logvar_xy_to_w = nn.Linear(self._hidden_dim, w_dim)

        # a seperate encoder is created, specifically for encoding the input data into the latent space
        # only here, you don't need the labels. This is basically the original VAE structure
        self.encoder_x_to_z = nn.LSTM(
            input_size=self._vocab_size,  # Input size only includes the data
            hidden_size=self._hidden_dim,
            num_layers=self._lstm_size,  # You can adjust the number of layers as needed
            batch_first=True,  # Batch dimension is first (batch_size, seq_len, input_size)
        )

        self.mu_x_to_z = nn.Linear(self._hidden_dim, self.z_dim)
        
        self.logvar_x_to_z = nn.Linear(self._hidden_dim, self.z_dim)
        
        self.encoder_y_to_w = nn.LSTM(
            input_size=self._labels_dim,  # Input size only includes the data
            hidden_size=self._hidden_dim,
            num_layers=self._lstm_size,  # You can adjust the number of layers as needed
            batch_first=True,  # Batch dimension is first (batch_size, seq_len, input_size)
        )

        self.mu_y_to_w = nn.Linear(self._hidden_dim, w_dim)
    
        self.logvar_y_to_w = nn.Linear(self._hidden_dim, w_dim)

        # decoder
        self.decoder_zw_to_x = nn.LSTM(
            input_size=self.z_dim+ w_dim,
            hidden_size=self._hidden_dim,  # Should match encoder hidden size
            num_layers=self._lstm_size,  # You can increase layers if needed
            batch_first=True,
            bidirectional=False  # Set to True for bidirectional LSTM
        )
        self.decoder_relu_zw_to_x = nn.ReLU()

        # Output layer
        self.mu_zw_to_x = nn.Linear(self._hidden_dim, self._vocab_size)
        self.logvar_zw_to_x = nn.Linear(self._hidden_dim, self._vocab_size)

        # decoder y
        self.decoder_z_to_y = nn.LSTM(
            input_size= self.z_dim,
            hidden_size=w_dim,  # Should match encoder hidden size
            num_layers=self._lstm_size,  # You can increase layers if needed
            batch_first=True,
            bidirectional=False  # Set to True for bidirectional LSTM
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.to(device)

    def q_zw(self, x, y):
        #encode x to z 
        x_to_z, _ = self.encoder_x_to_z(x)
        z_mu = self.mu_x_to_z(x_to_z) #use the last output
        z_logvar = self.logvar_x_to_z(x_to_z) #use the last output

        #encode xy to w
        xy = torch.cat([x, y], dim=2)
        xy_to_w, _ = self.encoder_xy_to_w(xy)
        w_mu_encoder = self.mu_xy_to_w(xy_to_w) #use the last output
        w_logvar_encoder = self.logvar_xy_to_w(xy_to_w) #use the last output

        # encode y to w
        y_to_w, _ = self.encoder_y_to_w(y)
        w_mu_prior = self.mu_y_to_w(y_to_w) #use the last output
        w_logvar_prior = self.logvar_y_to_w(y_to_w) #use the last output

        return (
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        )

    def p_x(self, z, w):
        zw = torch.cat([z, w], dim=2)
        output_decoder_zw,_ = self.decoder_zw_to_x(zw)
        output_decoder_zw = self.decoder_relu_zw_to_x(output_decoder_zw)
        mu = self.mu_zw_to_x(output_decoder_zw)
        logvar = self.logvar_zw_to_x(output_decoder_zw)
        return mu, logvar

    def forward(self, x, y):
        x = x.clone()

        (
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        ) = self.q_zw(x, y)

        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        z = self.reparameterize(z_mu, z_logvar)
        zw = torch.cat([z, w_encoder], dim=2)

        x_mu, x_logvar = self.p_x(z, w_encoder)
        y_pred, _ = self.decoder_z_to_y(z)

        return (
            x_mu,
            x_logvar,
            zw,
            y_pred,
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        )

    def predict(self, x, y):
        return self.forward(x, y)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    def fit(self, data, epochs=100, lr=1e-3, batch_size=32):
  
        x_train = data[0]

        if self._labels_dim == 2:
            y_data = data[1]
            y_prob_train = np.zeros((y_data.shape[0], 2))
            y_prob_train[:, 0] = 1 - y_data
            y_prob_train[:, 1] = y_data
        else:
            raise ValueError("Only binary class labels are implemented at the moment.")
        y_prob_train = np.repeat(y_prob_train[:, np.newaxis, :], x_train.shape[1], axis=1)
        train_loader = torch.utils.data.DataLoader(
            list(zip(x_train, y_prob_train)), shuffle=True, batch_size=batch_size
        )

        params_without_delta = [
            param
            for name, param in self.named_parameters()
            if "decoder_z_to_y" not in name
        ]
        params_delta = [
            param for name, param in self.named_parameters() if "decoder_z_to_y" in name
        ]

        opt_without_delta = optim.Adam(params_without_delta, lr=lr / 2)
        opt_delta = optim.Adam(params_delta, lr=lr / 2)

        train_x_recon_losses = []
        train_y_recon_losses = []

        log.info("Start training of CSVAE...")
        for i in trange(epochs):
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                (
                    loss_val,
                    x_recon_loss_val,
                    w_kl_loss_val,
                    z_kl_loss_val,
                    y_negentropy_loss_val,
                    y_recon_loss_val,
                ) = csvae_loss(self, x, y)

                opt_delta.zero_grad()
                y_recon_loss_val.backward(retain_graph=True)

                opt_without_delta.zero_grad()
                loss_val.backward()

                opt_without_delta.step()
                opt_delta.step()

                train_x_recon_losses.append(x_recon_loss_val.item())
                train_y_recon_losses.append(y_recon_loss_val.item())

            log.info(
                "epoch {}: x recon loss: {}".format(
                    i, np.mean(np.array(train_x_recon_losses))
                )
            )
            log.info(
                "epoch {}: y recon loss: {}".format(
                    i, np.mean(np.array(train_y_recon_losses))
                )
            )

        self.save()
        log.info("... finished training of CSVAE")

        self.eval()

    def save(self):
        cache_path = get_home()

        save_path = os.path.join(
            cache_path,
            "csvae_{}_{}.{}".format(self._data_name, self._input_dim, "pt"),
        )

        torch.save(self.state_dict(), save_path)

    def load(self, input_shape):
        cache_path = get_home()

        load_path = os.path.join(
            cache_path,
            "csvae_{}_{}.{}".format(self._data_name, input_shape, "pt"),
        )

        self.load_state_dict(torch.load(load_path))

        self.eval()

        return self
