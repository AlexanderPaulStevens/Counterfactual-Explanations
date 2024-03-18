# <span style="color:black">**REVISED: Generating Feasible and Plausible Counterfactual Explanations for Predictive Process Analytics**</span>

# 1. create_files.py

This file reads in the datasets in the folder `labeled_logs_csv_processed` and preprocessing, cleans and saves the files to the correct output files.

- `csv_files`
    - the `csv_files` folder contains the aggregation encoded data. 
- `dat_lab_files`
    - each row is a new trace (sequence) after labelencoding. This is used for the sequence mining algorithm IBCM.

- `hdf5_files`
    - this contains the onehotencoded (OHE) data to be used for the LSTM and the VAE. We use a hfd5_file due to its conventient writing and loading purposes.


# 2. IBCM_Python.py

This file reads the  `train.dat ` and  `train.lab ` files of the  `csv_files` folder. It extracts the Declare patterns and saves them to `IBCM\dataset_name\`


# 3. hyperopt_VAE_GC.ipynb and hyperopt_LSTM.py

These two files are used to find the optimal hyperparameters for the LSTM and the LSTM-VAE.

The parameters of the LSTM model are:

    - dropout: float, default: 0.1
        - Dropout rate. 

    - hidden_sizes: list[int]
        - hidden_sizes[0] contains the first  hidden size
        - hidden_sizes[1] contains the second hidden size
        - hidden_sizes[2] contains the number of lstm layers
    
    - learning_rate: float
        - Learning rate for the training.

The parameters for the LSTM VAE are run with WandB to have more control over the learning capabilities and loss function. You can find the parameters in `hyperoptimization\config\VAE_sweep.yaml`

The parameters of the (LSTM) VAE model are:

    - batch_size: int
        - Number of samples in each batch

    - hidden_dim
        - contains the hidden size
    
    - latent dim
        - contains the latent dim
    
    - lstm_size
        - contains the number of lstm layers

    - learning_rate: float
        - Learning rate for the training.


### For the VAE:
#### for following datasets, we have ran the `joint_constraints_in_loss` file:

- sepsis_cases_2
- sepsis_cases_4
- bpic2012_accepted
- bpic2012_declined

#### for following datasets, we have ran the `no_constraints_in_loss` file:

- sepsis_cases_2
- bpic2012_accepted

### For the LSTM:
#### for following datasets, we have completed the hyperparameter search:

- sepsis_cases_1
- sepsis_cases_2
- sepsis_cases_4
- bpic2012_accepted

# 4. Experiment_VAE.py and experiment_LSTM.py

These files loop over the saved models and extract the best model.

- experiment_LSTM is selected based on the highest AUC
- experiment_VAE is selected based on the minimal loss


