import json


def load(config_path: str) -> dict:
    """Load hyperparameters from JSON file."""
    default_config = {
        'noise_dim': 100,
        'signal_length': 5000,
        'generator_initial_dense': 1250,
        'generator_output_kernel': 16,
        'generator_conv_layers': [
            {'filters': 64, 'kernel_size': 16, 'strides': 2},
            {'filters': 32, 'kernel_size': 16, 'strides': 2}
        ],
        'critic_branch1': {
            'filters': 32,
            'kernel_size': 16,
            'pool_size': 4
        },
        'critic_branch2': {
            'filters': 32,
            'kernel_size': 16,
            'pool_size': 4
        },
        'critic_final_conv': {
            'filters': 64,
            'kernel_size': 8
        }
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
