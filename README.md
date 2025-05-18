# Generating Feasible and Plausible Counterfactual Explanations for Outcome Prediction of Business Processes

Link to ArXiv: https://arxiv.org/abs/2403.09232

<img width="1000" alt="REVISED-table" src="https://github.com/AlexanderPaulStevens/Counterfactual-Explanations/assets/75080516/7fa9e9c2-1a25-4181-b2e6-c591314a2b72">

# Hyperparameter settings

## Model Architecture
- **Hidden Dimension**: 200 (default)
- **Latent Size**: Configurable through layers[1]
- **LSTM Layers**: 5 (default)
- **Bidirectional**: False
- **Dropout**: 0.3 (default)

## Training Parameters
- **Learning Rate**: 5e-5 (default)
- **Epochs**: 500 (default)
- **Batch Size**: 128 (default)
- **KL Weight**: 0.3 (default)
- **Lambda Regularization**: 1e-6 (default)
- **Gradient Clipping**: max_norm = 5.0

## Loss Components
- **Reconstruction Loss**: Cross Entropy Loss
- **KL Divergence Loss**: Standard VAE KL divergence
- **Joint Constraint Loss**: Optional, controlled by `joint_constraint_in_loss` parameter

## Optimizer
- **Type**: Adam
- **Weight Decay**: 1e-6 (lambda_reg)

## Model Components
- **Encoder**: LSTM
- **Decoder**: LSTM
- **Latent Space**: Gaussian distribution
- **Output Layer**: Linear layer with LogSoftmax

## Additional Notes
- The model supports both CPU and GPU training (automatically selects device)
- Xavier initialization is used for linear layers
- The model includes padding handling for variable length sequences
- Supports both constrained and unconstrained training modes 
