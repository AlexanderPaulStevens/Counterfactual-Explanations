import numpy as np
import tensorflow as tf
import torch
import torch.distributions as dists
from keras import backend as K
from torch import nn


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


def csvae_loss(csvae, x_train, y_train):
    #this function computes the loss components for trainig a CSVAE and constructs the overall loss (ELBO)
    x = x_train.clone().float()
    y = y_train.clone().float()

    (
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
    ) = csvae.forward(x, y)

    #the reconstruction loss (in MSE)
    x_recon = nn.MSELoss()(x_mu, x)

    # The KL divergence losses are calculated for both the latent space (z) and the class-related latent variables (w)
    # these losses measure how the distributions of these variables diverge from predefined priors
    w_dist = dists.MultivariateNormal(
        w_mu_encoder.flatten(), torch.diag(w_logvar_encoder.flatten().exp())
    )
    w_prior = dists.MultivariateNormal(
        w_mu_prior.flatten(), torch.diag(w_logvar_prior.flatten().exp())
    )
    # w_kl calculates the divergence between the posterior distribution 'w' and a prior distribution
    w_kl = dists.kl.kl_divergence(w_dist, w_prior)

    z_dist = dists.MultivariateNormal(
        z_mu.flatten(), torch.diag(z_logvar.flatten().exp())
    )

    # this is the prior distribution for the latent space variables 'z'
    # torch eye creates an identity matrix with size z_dim x z_dim
    # The identity matrix represents a covariance matrix where all off-diagonal elements are zero, 
    # and diagonal elements are one, assuming unit variance along each dimension.

    # Define the size of the 3D tensor
    z_prior = dists.MultivariateNormal(
        torch.zeros(z_mu.size(0)*z_mu.size(1)*z_mu.size(2), device=z_mu.device),
        torch.eye(z_mu.size(0)*z_mu.size(1)*z_mu.size(2), device=z_mu.device),
    )
    
    # z_kl calculates the divergence between the posterior distribution 'z' and a prior distribution
    z_kl = dists.kl.kl_divergence(z_dist, z_prior)

    y_pred_avg = y_pred[:,-1,:]
    y_pred_negentropy = (
        y_pred.log() * y_pred + (1 - y_pred).log() * (1 - y_pred)
    ).mean()

    class_label = torch.argmax(y, dim=1)
    y_recon = (
        100.0
        * torch.where(
            class_label == 1, -torch.log(y_pred[:, 1]), -torch.log(y_pred[:, 0])
        )
    ).mean()

    ELBO = 40 * x_recon + 0.2 * z_kl + 1 * w_kl + 110 * y_pred_negentropy

    return ELBO, x_recon, w_kl, z_kl, y_pred_negentropy, y_recon
