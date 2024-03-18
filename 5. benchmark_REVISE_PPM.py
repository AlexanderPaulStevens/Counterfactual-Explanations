import warnings
warnings.filterwarnings("ignore") #ignore futurewarnings
from carla import Benchmark
import carla.evaluation.catalog as evaluation_catalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog
from carla.data.catalog.own_catalog import OwnCatalog
import numpy as np
import torch
import pandas as pd
import os
import re
from util.settings import global_setting, training_setting
from util.DatasetManager import DatasetManager
from carla.iBCM.check_constraint_violation import ProcessConstraints
from collections import Counter

target = [0]
threshold = 0.5
mode = 'test' # if we want to generate counterfactuals for the test data
max_iter = 1500 # the number of iterations for the optimization
joint_constraint_in_loss = True # if we want to use the joint constraint in the loss function

dataset_names = ['production','sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4', 
                 'bpic2012_accepted', 'bpic2012_declined', 'bpic2012_cancelled', 
                 'bpic2015_1_f2','bpic2015_2_f2','bpic2015_3_f2','bpic2015_4_f2','bpic2015_5_f2']
dataset_name = 'production' 
dataset_manager = DatasetManager(dataset_name)

print('own dataset', dataset_name)
dataset = OwnCatalog(dataset_name)

# To get the training or testing dataset, use df, df_train or df_test. These are the @property that are defined in the class MyOwnDataSet
data_train = dataset.df_train
data_test = dataset.df_test

# check whether every preprocessing step went well
# reversed_activity_train = dataset_manager.reverse_ohe_to_padded_activity(np.array(data_train))

#the list of labels for the test and training sequences
label_train = dataset.target_train
label_test = dataset.target_test

#we make a dictionary of the column names. This will help us in the future to get the original activities back.
column_names = dataset._column_names
column_names_dict = {}
for index, element in enumerate(column_names):
    column_names_dict[index] = element

# Get the shape of the tensor
print('The type of the dataset is:', type(data_test), type(data_train))
print('The training data contains', data_train.shape[0], 'instances and', data_train.shape[1], 'sequence length and vocab size', data_train.shape[2])
print('The testing data contains', data_test.shape[0], 'instances and', data_test.shape[1], 'sequence length and vocab size', data_test.shape[2])
print('The training and testing label shape', label_train.shape, label_test.shape)
print('The number of positive labels in the training set', np.sum(label_train==0))
print('The number of negative labels in the training set', np.sum(label_train==1))
print('The number of positive labels in the test set', np.sum(label_test==0))
print('The number of negative labels in the test set', np.sum(label_test==1))

# to extract the best parameters:
best_LSTMs = global_setting['best_LSTMs']
if dataset_name in ['sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4']:
    dir_path = best_LSTMs + 'sepsis_cases_2'
elif dataset_name in ['bpic2012_accepted', 'bpic2012_declined', 'bpic2012_cancelled']:
    dir_path = best_LSTMs + 'sepsis_cases_2'
else:
    dir_path = best_LSTMs + 'sepsis_cases_2'
dir_list = os.listdir(dir_path)

if 'desktop.ini' in dir_list:
    dir_list.remove('desktop.ini')

for LSTM_name in dir_list:
    print(LSTM_name)
    split = LSTM_name.split("_")
    checkpoint = torch.load(dir_path+'/'+LSTM_name, map_location=torch.device('cpu'))
    LSTM_lr = float(split[1])
    LSTM_hidden_dim1 = int(split[2])
    LSTM_hidden_dim2 = int(split[3])
    lSTM_size = int(split[4])
    LSTM_dropout = float(split[5])
    LSTM_epoch = [int(match) for match in re.findall(r'\d+', str(split[6]))][0]
    print('epoch', LSTM_epoch)

# the training parameters
LSTM_lr = 0.001
LSTM_hidden_dim1 = 50
LSTM_hidden_dim2 = 20
lSTM_size = 2
LSTM_dropout = 0.3
LSTM_epoch = 100

training_params = {"optimizer_name": "Adam", "lr": LSTM_lr, "weight_decay":1e-5, "epochs": LSTM_epoch, "hidden_size": [LSTM_hidden_dim1, LSTM_hidden_dim2, lSTM_size], "dropout": LSTM_dropout, "batch_size": 1}

ml_model = MLModelCatalog(
    dataset,
    model_type="lstm",
    load_online=False,
    backend="pytorch"
)

ml_model.train(
    dataset_name=dataset_name,
    dataset_manager=dataset_manager,
    optimizer_name= training_params['optimizer_name'],
    learning_rate=training_params["lr"],
    weight_decay = training_params["weight_decay"],
    epochs=training_params["epochs"],
    batch_size=training_params["batch_size"], #minibatch so you can take the last hidden state that is not a padding token
    hidden_sizes=training_params["hidden_size"],
    dropout = training_params["dropout"],
    vocab_size = dataset._vocab_size,
    force_train=False)

# Print the values of the properties. You first need to train (or load) it obviously
print("Activity values (ordered)", ml_model.feature_input_order)
print("Raw Model:", ml_model.raw_model)
print('The ML model training data contains', ml_model.data.df_train.shape[0], 'instances and', ml_model.data.df_train.shape[1], 'sequence length and vocab size of', ml_model.data.df_train.shape[2])
print('The ML model testing data contains', ml_model.data.df_test.shape[0], 'instances and', ml_model.data.df_test.shape[1], 'sequence length and vocab size of', ml_model.data.df_test.shape[2])

# Count the occurrences of each value
value_counts = Counter(dataset.target_test)

# Print the counts for each value
for value, count in value_counts.items():
    print(f"{value}: {count} times for the test data")

# these contain the negative factuals that were correctly predicted to be negative (i.e. class 1 in the case of sepsis)
negative_factuals = predict_negative_instances(ml_model, dataset, threshold, mode, target)
print('the factuals', negative_factuals.shape)

#search for traces where there are any constraint violations
constraintminer = ProcessConstraints(dataset_name, dataset.vocab_size, target) # we should always calculate the violations, even if we do not use them in our loss function

# This was to select some factuals and do some EDA
# selected_negative_factuals = constraintminer.find_violated_traces(negative_factuals, constraint_list = 'total', verbose=False)
# print('the negative factuals that have violated constraints', selected_negative_factuals.shape)
# size_of_queries = 10
# random_indices = torch.tensor([2,10,15,23,56,76,88,130,132,140]) # Generate random indices for batch selection
# random_indices = torch.randint(0, selected_negative_factuals.shape[0], size= (size_of_queries,))  # Select random batches
# From here, I would do a loop and generate a dataframe per query. Then you can pd.concat() all of them together
# selected_batches = selected_negative_factuals[random_indices]

print('the selected counterfactual:', torch.argmax(negative_factuals, axis=2))
print('the selected factual that has shape', negative_factuals.shape)

if joint_constraint_in_loss:
    #best_VAEs = global_setting['best_VAEs']
    #dir_path = best_VAEs + dataset_name
    #dir_list = os.listdir(dir_path)
    outfile = "results/joint_constraints/"
    dataset_manager.ensure_path(outfile)
    outfile += "results_" + dataset_name + "_" + str(max_iter) + "_" + "joint_constraints.csv"
else:
    #best_VAEs_no_PC = global_setting['best_VAEs_no_PC']
    #dir_path = best_VAEs_no_PC + dataset_name
    #dir_list = os.listdir(dir_path)
    outfile = "results/no_constraints/"
    dataset_manager.ensure_path(outfile)
    outfile += "results_" + dataset_name + "_" + str(max_iter) + "_" + "no_constraints.csv"

# this is based on the insights obtained from WandB
if dataset_name in ['sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4']:
    VAE_lr = 0.001
    VAE_hidden_dim = 100
    VAE_latent_dim = 50
    VAE_lstm_size = 3

# this is based on the insights obtained from WandB
if dataset_name in ['bpic2012_accepted', 'bpic2012_cancelled', 'bpic2012_declined']:
    VAE_lr = 0.001
    VAE_hidden_dim = 100
    VAE_latent_dim = 50
    VAE_lstm_size = 3

if dataset_name in ['production']:
    VAE_lr = 0.05
    VAE_hidden_dim = 100
    VAE_latent_dim = 50
    VAE_lstm_size = 1

if dataset_name in ['bpic2015_1_f2','bpic2015_2_f2','bpic2015_3_f2','bpic2015_4_f2','bpic2015_5_f2']:
    VAE_lr = 0.001
    VAE_hidden_dim = 100
    VAE_latent_dim = 30
    VAE_lstm_size = 1

"""
If we look at the hyperparams for REVISE (CARLA benchmark)
only 10 or 25 epochs (100 for the big dataset)
    - batch sizes 128, 1024, 2048 (big dataset)
    - 1500 iterations based on their grid search
    - learning rate: 0.1
    - similarity weight lambda: 0.5
"""
hyperparams = {
    "data_name": dataset.name, #name of the dataset
    "lambdas": [20, 2 ,10], # decides how similar the counterfactual is to the factual
    "optimizer": "adam", #adam, or RMSprop (optimizer for the generation of counterfactuals)
    "lr": 0.1, # learning rate for Revise
    "max_iter": max_iter,  #number of iterations for Revise optimization
    "target_class": target, # target class
    "vocab_size": dataset.vocab_size, # the vocabulary size of the activitiew
    "max_prefix_length": dataset.max_prefix_length, #the maximum sequence length
    "threshold": 0.5, # the threshold of probability before you consider the predicted to be flipped
    "loss_diff_threshold": 1e-5, # the loss difference threshold (0.00001), 
                                 # if less than threshold, a stopping threshold is triggered
    "vae_params": { # the vae params is a dictionary with the learning parameters for the VAE
        "layers": [VAE_hidden_dim, VAE_latent_dim, VAE_lstm_size], # hidden size, latent size, lstm size of the VAE
        "train": True, # force VAE training or not
        "joint_constraint": joint_constraint_in_loss,
        "lambda_reg": 1e-5, # this is the lambda for the optimizer of the VAE
        "epochs": training_setting['epochs_VAE'], # the epochs for VAE training
        "lr": VAE_lr, # learning rate optimizer VAE
    },
}

counterfactual_alg = recourse_catalog.ReviseOriginal(ml_model, dataset, hyperparams, constraintminer)

#df_cfs = counterfactual_alg.get_counterfactuals(selected_batches)
#print(df_cfs)

# first initialize the benchmarking class by passing
# black-box-model, recourse method, and factuals into it
print('the benchmark has started')
number = 0

# Create an empty DataFrame to store the results
all_results = pd.DataFrame()

for selected_batch in negative_factuals:
    print('selected batch', number)
    selected_batch = selected_batch.unsqueeze(0)
    benchmark = Benchmark(dataset_manager, ml_model, counterfactual_alg, selected_batch, hyperparams['threshold'])

    # now you can decide if you want to run all measurements
    # or just specific ones.
    evaluation_measures = [
        evaluation_catalog.YNN(dataset_manager, benchmark.mlmodel, {"y": 5, "cf_label": target[0], "NN": 5}),
        evaluation_catalog.Distance(dataset_manager, benchmark.mlmodel),
        evaluation_catalog.ConstraintViolation(dataset_manager, benchmark.mlmodel, constraintminer)
    ]

    # now run all implemented measurements and concatenate the results
    print('The results:')
    results = benchmark.run_benchmark(evaluation_measures)
    all_results = pd.concat([all_results, results], ignore_index=True)

    print(results.head(20))
    number +=1

all_results.to_csv(outfile, index=False)
