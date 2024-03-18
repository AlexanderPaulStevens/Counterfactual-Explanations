global_setting = {
    "train_ratio": 0.8,
    "train_val_ratio": 0.8,
    "seed": 42,
    "home_directory":   '',

    "params_dir_LSTM": 'carla/hyperoptimization/params_dir_LSTM/',
    "hyper_models":    'carla/hyperoptimization/params_dir_LSTM/hyper_models/',
    "best_LSTMs":      'carla/hyperoptimization/params_dir_LSTM/best_models/',

    "params_dir_VAE":  'carla/hyperoptimization/params_dir_VAE/',
    "hyper_VAEs":      'carla/hyperoptimization/params_dir_VAE/hyper_VAEs/',
    "best_VAEs":       'carla/hyperoptimization/params_dir_VAE/best_VAEs/',

    "params_dir_VAE_no_PC":  'carla/hyperoptimization/params_dir_VAE_no_PC/',
    "hyper_VAEs_no_PC":      'carla/hyperoptimization/params_dir_VAE_no_PC/hyper_VAEs_no_PC/',
    "best_VAEs_no_PC":       'carla/hyperoptimization/params_dir_VAE_no_PC/best_VAEs_no_PC/',

    "n_splits": 3,
    "max_evals": 10,
}

model_setting = {
    "lstm_layer": 1
}

training_setting = {
    "epochs_VAE": 250,
    "epochs_LSTM": 100,
    "bptt": 60,
    "clip": 0.25,
    "train_losses": [],
    "test_losses": [],
    "patience": 10,
    "embed_size": 32
}
