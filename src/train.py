from network import *


generator = create_generator()
critic = create_critic()
X_real = np.load('../data/data.npy')
train_wgan_gp(generator, critic, X_real, epochs=200, batch_size=32)
