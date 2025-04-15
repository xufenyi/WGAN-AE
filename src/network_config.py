import json


def load(config_path: str) -> dict:
    """Load hyperparameters from JSON file."""
    default_config = {
        "noise_dim": 100,
        "signal_length": 5000,
        "generator_layers": [
            {
                "name": "Dense",
                "units": 1000
            },
            {
                "name": "BatchNormalization"
            },
            {
                "name": "Dense",
                "units": 5000
            },
            {
                "name": "BatchNormalization"
            },
            {
                "name": "Dense",
                "units": 10000
            },
            {
                "name": "BatchNormalization"
            },
            {
                "name": "Reshape",
                "target_shape": [1250, 8]
            },
            {
                "name": "Conv1DTranspose",
                "filters": 32,
                "kernel_size": 8,
                "strides": 2
            },
            {
                "name": "BatchNormalization"
            },
            {
                "name": "Conv1DTranspose",
                "filters": 32,
                "kernel_size": 8,
                "strides": 2
            },
            {
                "name": "BatchNormalization"
            },
            {
                "name": "Conv1D",
                "filters": 1,
                "kernel_size": 16,
                "activation": "tanh"
            }
        ],
        "critic_branch1_layers": [
            {
                "name": "Conv1D",
                "filters": 16,
                "kernel_size": 32,
                "strides": 4,
                "padding": "valid"
            },
            {
                "name": "Conv1D",
                "filters": 32,
                "kernel_size": 16,
                "strides": 1
            },
            {
                "name": "MaxPooling1D",
                "pool_size": 4,
                "strides": 2
            }
        ],
        "critic_branch2_layers": [
            {
                "name": "Conv1D",
                "filters": 16,
                "kernel_size": 32,
                "strides": 4,
                "padding": "valid"
            },
            {
                "name": "Conv1D",
                "filters": 32,
                "kernel_size": 16,
                "strides": 1
            },
            {
                "name": "MaxPooling1D",
                "pool_size": 4,
                "strides": 2
            }
        ],
        "critic_merged_layers": [
            {
                "name": "Conv1D",
                "filters": 16,
                "kernel_size": 16
            },
            {
                "name": "Dropout",
                "rate": 0.05
            },
            {
                "name": "Conv1D",
                "filters": 32,
                "kernel_size": 8
            },
            {
                "name": "Dropout",
                "rate": 0.05
            },
            {
                "name": "Conv1D",
                "filters": 32,
                "kernel_size": 4
            },
            {
                "name": "GlobalAveragePooling1D"
            }
        ]
    }

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
    except Exception as e:
        print(f'Error loading config file: {str(e)}')
        print('Using default configuration.')
        return default_config

    config = default_config.copy()
    config.update(loaded_config)
    return config
