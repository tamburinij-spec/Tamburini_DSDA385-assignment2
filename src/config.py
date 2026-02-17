import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
MODEL_CONFIG = {
    'num_classes': 10,
    'input_channels': 3,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'weight_decay': 1e-5,
}

# Data configuration
DATA_CONFIG = {
    'data_dir': './data',
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
}

# Paths
PATHS = {
    'models_dir': './models',
    'logs_dir': './logs',
    'results_dir': './results',
}