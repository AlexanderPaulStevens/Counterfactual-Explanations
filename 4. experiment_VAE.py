# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:07:08 2023

@author: u0138175
"""

####################PACKAGES AND FUNCTIONS#######################
import torch
from torch.utils.data import DataLoader, TensorDataset
import re
import os
import sys
import numpy as np
import random
import warnings
# set logging
import logging
logging.getLogger().setLevel(logging.INFO)
sys.path.append(os.getcwd())
from util.DatasetManager import DatasetManager
from util.settings import global_setting, training_setting
from carla.data.catalog.own_catalog import OwnCatalog
from carla.recourse_methods.autoencoder.models import VariationalAutoencoder, CheckpointSaver
from carla.iBCM.check_constraint_violation import ProcessConstraints
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(22)
#####################PARAMETERS###################################
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

dataset_names = ['production','sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4', 'bpic2012_accepted', 'bpic2012_rejected', 'bpic2012_cancelled']
dataset_name = dataset_names[2]
print('own dataset', dataset_name)
dataset_manager = DatasetManager(dataset_name)
dataset = OwnCatalog(dataset_name)

best_loss = 999
best_VAE_name = ""
beta = 0.3
beta2 = 1
joint_constraint_in_loss = False #CHANGE THIS


g = torch.Generator()
g.manual_seed(0)

target = [0]
epochs = training_setting['epochs_VAE']

recon_loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)

#the test data
test = dataset.df_train

if joint_constraint_in_loss==True:
    #search for traces where there are any constraint violations
    constraintminer = ProcessConstraints(dataset_name,dataset.vocab_size, target)
    #paths
    path = 'carla/hyperoptimization/params_dir_VAE'
    dir_path = path+'/hyper_VAEs/'+dataset_name
    project_name = 'counterfactuals_LSTMVAE'
    best_VAEs = global_setting['best_VAEs']
else:
    constraintminer = None
    path = 'carla/hyperoptimization/params_dir_VAE_no_PC'
    dir_path = path+'/hyper_VAEs_no_PC/'+dataset_name
    project_name = 'counterfactuals_LSTMVAE_no_PC'
    best_VAEs = global_setting['best_VAEs_no_PC']

dataset_manager.ensure_path(path)
dataset_manager.ensure_path(dir_path)
dataset_manager.ensure_path(best_VAEs)


dir_list = os.listdir(dir_path)
if 'desktop.ini' in dir_list:
    dir_list.remove('desktop.ini')

if 'wandb' in dir_list:
    dir_list.remove('wandb')

print('dir list', dir_list)

for VAE_name in dir_list:
    print(VAE_name)
    print(best_loss)
    split = VAE_name.split("_")
    checkpoint = torch.load(dir_path+'/'+VAE_name, map_location=torch.device('cpu'))
    learning_rate = float(split[1])
    hidden_dim = int(split[2])
    latent_dim = int(split[3])
    lstm_size = int(split[4])
    optimizer = int(split[6])
    print('epoch', str(split[6]))
    epoch = [int(match) for match in re.findall(r'\d+', str(split[6]))]
    model = VariationalAutoencoder(dataset_name, [hidden_dim, latent_dim, lstm_size], dataset.vocab_size, dataset.max_prefix_length, epoch, constraintminer, joint_constraint_in_loss)
    model.load_state_dict(checkpoint)
    model.eval()
    padding_tensor = torch.zeros([dataset.vocab_size])
    padding_tensor[0] =1
    mask_test = ~torch.all(torch.eq(test, padding_tensor), dim=2)
    test = test * mask_test.unsqueeze(-1).float()  # Apply the mask to ignore padded sequences
    reconstruction_test, mu_test, log_var_test = model(test) # returns the reconstruction, mu and log_var
    reconstruction_test  = model.mask_out_tensor(reconstruction_test)
    reconstruction_constraint_test = reconstruction_test.clone().detach()
    class_indices_test = torch.argmax(test, dim=2)
    class_indices_test = class_indices_test.view(-1)
    reconstruction_test = reconstruction_test.view(-1, dataset.vocab_size)
    recon_loss_test = recon_loss_function(reconstruction_test, class_indices_test)
    kld_loss_test = model.kld(mu_test, log_var_test)
    if joint_constraint_in_loss==True:
        reconstruction_constraint = reconstruction_test.clone().detach()
        constraint_loss_test = model.constraint_violation.calculate_joint_constraint_loss(reconstruction_constraint_test)
        validation_loss = recon_loss_test + beta * kld_loss_test + beta2 * constraint_loss_test
    else:
        validation_loss = recon_loss_test + beta * kld_loss_test
    print('validation loss', validation_loss)
    if validation_loss < best_loss:
        print('loss',validation_loss,'best loss', best_loss)
        best_loss = validation_loss
        best_model = model
        best_VAE_name = VAE_name
        print('best loss now is:', best_loss)
print('best model with name:', best_VAE_name, 'and loss', best_loss)
path_data_label = best_VAEs+'/' + dataset_name+'/'
if not os.path.exists(os.path.join(path_data_label)):
    os.makedirs(os.path.join(path_data_label))
best_model_path = os.path.join(path_data_label, best_VAE_name)
torch.save(best_model.state_dict(), best_model_path)