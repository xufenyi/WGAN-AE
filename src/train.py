import numpy as np
from network import create_generator, create_critic, train_wgan_gp
import network_config


config = network_config.load('./network_config.json')
generator = create_generator(config=config)
critic = create_critic(config=config)
X_real = np.load('../data/data.npy')
train_wgan_gp(generator, critic, X_real, epochs=200, batch_size=32)
