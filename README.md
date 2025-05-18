# Generating Feasible and Plausible Counterfactual Explanations for Outcome Prediction of Business Processes

Link to ArXiv: https://arxiv.org/abs/2403.09232

<img width="1000" alt="REVISED-table" src="https://github.com/AlexanderPaulStevens/Counterfactual-Explanations/assets/75080516/7fa9e9c2-1a25-4181-b2e6-c591314a2b72">

# Hyperparameter settings (LSTM predictive model)

This section outlines the hyperparameters used for training the LSTM predictive model.

## Model Architecture

The LSTM model consists of:
- Bidirectional LSTM layer
- Two fully connected layers
- Sigmoid activation for final output

### Architecture Parameters
- **Input Size**: Vocabulary size of the dataset
- **LSTM Layer**:
  - Hidden Dimension 1 (hidden_dim1)
  - Hidden Dimension 2 (hidden_dim2)
  - Number of LSTM layers (lstm_size)
  - Dropout rate
  - Bidirectional: True

## Training Parameters

- **Optimizer**: Adam
- **Loss Function**: Binary Cross Entropy (implicit through sigmoid output)
- **Evaluation Metric**: ROC AUC Score
- **Cross Validation**: Stratified K-Fold (n_splits = 3)
- **Weight Decay**: 1e-5
- **Gradient Clipping**: 0.25
- **Batch Size**: 128
- **Epochs**: 100 (default from training settings)

## Hyperparameter Search Space

The following ranges were used for hyperparameter optimization:
- Dropout: [0.1, 0.2, 0.3]
- Hidden Dimension 1: [50, 100]
- Hidden Dimension 2: [10, 50]
- LSTM Size: [2, 5]
- Learning Rate: [0.001, 0.002, 0.005, 0.01]

## Dataset-Specific Parameters

### BPIC 2012 Datasets (accepted, cancelled, declined)
- Learning Rate: 0.001
- Hidden Dimension 1: 50
- Hidden Dimension 2: 20
- LSTM Size: 2
- Dropout: 0.3

### Production Dataset
- Learning Rate: 0.001
- Hidden Dimension 1: 100
- Hidden Dimension 2: 50
- LSTM Size: 2
- Dropout: 0.2

### Sepsis Cases Datasets
- Learning Rate: 0.001
- Hidden Dimension 1: 100
- Hidden Dimension 2: 50
- LSTM Size: 2
- Dropout: 0.2

## Additional Notes

- The model uses sequence length masking to handle variable-length sequences
- For training and validation, it uses the last non-padding token's hidden state
- For inference, it uses the last hidden state of the sequence
- The model includes a checkpoint saver that keeps track of the top N best models based on validation metric
- Early stopping is implemented with:
  - Early start patience: 15 epochs
  - Early stop patience: 60 epochs
- Learning rate reduction is implemented with:
  - Factor: 0.5
  - Patience: 10 epochs
  - Threshold: 0.0001
  - Min learning rate: 0 

# Hyperparameter settings (LSTM variational autoencoder)

This section outlines the hyperparameters used for training the LSTM Variational Autoencoder (LSTM VAE) in the project.

## General Training Parameters

- **Optimizer**: Adam
- **Weight Decay (lambda_reg)**: 1e-6
- **KL Weight**: 0.3 (linearly increases during training)
- **Gradient Clipping**: 5.0
- **Batch Size**: 128
- **Epochs**: 250 (default from training settings)

## Architecture Parameters

The VAE architecture consists of:
- LSTM-based encoder and decoder
- Hidden dimension (hidden_dim)
- Latent dimension (latent_dim)
- Number of LSTM layers (lstm_size)

## Dataset-Specific Parameters

### BPIC 2012 Datasets (accepted, cancelled, declined)
- Learning Rate: 0.001
- Hidden Dimension: 100
- Latent Dimension: 50
- LSTM Size: 3

### Production Dataset
- Learning Rate: 0.05
- Hidden Dimension: 100
- Latent Dimension: 50
- LSTM Size: 1

### BPIC 2015 Datasets
- Learning Rate: 0.001
- Hidden Dimension: 100
- Latent Dimension: 30
- LSTM Size: 1

## Hyperparameter Search Space

The following ranges were used for hyperparameter optimization:
- Hidden Dimension: [50, 100]
- Latent Dimension: [10, 30, 50]
- LSTM Size: [1, 2, 3, 4, 5]
- Learning Rate: [0.1, 0.01, 0.05, 0.001]

## Additional Notes

- The VAE uses a reparameterization trick for sampling from the latent space
- The model includes both reconstruction loss and KL divergence loss
- The model uses cross-entropy loss for reconstruction
- Early stopping is implemented with patience of 10 epochs
- No differences in the training hyperparameters between the two settings, i.e. whether or not constraints were included in the loss function
