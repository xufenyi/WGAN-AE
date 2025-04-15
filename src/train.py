import numpy as np
from network import create_generator, create_critic, train_wgan_gp
import network_config


config = network_config.load('./network_config.json')
generator = create_generator(config=config)
critic = create_critic(config=config)
X_real = np.load('../data/data.npy')
train_wgan_gp(generator, critic, X_real, epochs=250, batch_size=32)

noise = np.random.randn(1, config['noise_dim'])
fake_signal = generator.predict(noise)[0]
with open('fake_signal.csv', 'w', encoding='utf-8') as f:
    for i in range(len(fake_signal)):
        f.write(f'{str(i)},{str(fake_signal[i][0])}\n')