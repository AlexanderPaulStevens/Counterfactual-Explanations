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
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import roc_auc_score
import logging # set logging
logging.getLogger().setLevel(logging.INFO)
sys.path.append(os.getcwd())
from util.DatasetManager import DatasetManager
from util.settings import global_setting, training_setting
from carla.data.catalog.own_catalog import OwnCatalog
from carla.models.catalog.LSTM_TORCH import LSTMModel, CheckpointSaver
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
torch.autograd.set_detect_anomaly(True) # to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.manual_seed(22)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
#####################PARAMETERS###################################

dataset_names = ['production','sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4', 'bpic2012_accepted', 'bpic2012_rejected', 'bpic2012_cancelled']
dataset_name = 'sepsis_cases_2'
print('own dataset', dataset_name)
dataset = OwnCatalog(dataset_name)
train_ratio = global_setting['train_ratio']

dataset_manager = DatasetManager(dataset_name)

best_LSTMs = global_setting['best_LSTMs']
dir_path = global_setting['hyper_models'] + dataset_name
dataset_manager.ensure_path(best_LSTMs)
dataset_manager.ensure_path(dir_path)
dir_list = os.listdir(dir_path)

n_splits = global_setting['n_splits']
max_evals = global_setting['max_evals']
train_ratio = global_setting['train_ratio']

# we only use the training dataset for hyperoptimization
data_train = dataset.df_train
target_train = dataset.target_train

skf = StratifiedKFold(n_splits=n_splits)
skf.get_n_splits(data_train, target_train)

data_cv = []
target_cv = []
for i, (train_index, val_index) in enumerate(skf.split(data_train, target_train)):
    data_cv.append(data_train[val_index]) # you add all the val chunks
    target_cv.append(target_train[val_index])

# remove the redundant files
if 'desktop.ini' in dir_list:
    dir_list.remove('desktop.ini')

best_auc = 0.5
best_LSTM_name = ""
for LSTM_name in dir_list:
    print('the current LSTM name', LSTM_name)
    print('the current best AUC is', best_auc)
    split = LSTM_name.split("_")
    checkpoint = torch.load(dir_path+'/'+LSTM_name, map_location=torch.device('cpu'))
    LSTM_lr = float(split[1])
    LSTM_hidden_dim1 = int(split[2])
    LSTM_hidden_dim2 = int(split[3])
    lSTM_size = int(split[4])
    LSTM_dropout = float(split[5])
    LSTM_epoch = [int(match) for match in re.findall(r'\d+', str(split[6]))][0]
    print('epoch', LSTM_epoch)
    model = LSTMModel(dataset.vocab_size, LSTM_dropout, [LSTM_hidden_dim1, LSTM_hidden_dim2, lSTM_size])
    model.load_state_dict(checkpoint)
    model.eval()

    # Perform stratified k-fold cross-validation
    all_preds = []
    all_targets = []
    auc_list = []
    for i, (train_index, val_index) in enumerate(skf.split(data_train, target_train)):
        val_data = data_train[val_index]
        val_target = target_train[val_index]
        pred = model(val_data, mode='val').squeeze(-1).to('cpu').detach().numpy()
        auc = roc_auc_score(val_target, pred)
        auc_list.append(auc)
        print('here',auc)
    average_AUC = np.mean(auc_list)
    print('average_AUC',average_AUC)
    if average_AUC > best_auc:
        print('auc',average_AUC,'best_auc', best_auc)
        best_auc = average_AUC.copy()
        best_model = model
        best_LSTM_name = LSTM_name
        print('best auc now is:', best_auc)
print('best model with name:', best_LSTM_name, 'and auc', best_auc)
path_data_label = best_LSTMs+'/' + dataset_name+'/'
if not os.path.exists(os.path.join(path_data_label)):
    os.makedirs(os.path.join(path_data_label))
best_model_path = os.path.join(path_data_label, best_LSTM_name)
torch.save(best_model.state_dict(), best_model_path)