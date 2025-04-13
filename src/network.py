import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
from datetime import datetime


class FFTPreprocessLayer(layers.Layer):
    """
    A custom Keras layer that applies Fast Fourier Transform (FFT) to the input signal.
    """
    def __init__(self, **kwargs):
        super(FFTPreprocessLayer, self).__init__(**kwargs)
        self.trainable = False

    def call(self, inputs, training=None, **kwargs):
        # Parameters `training` and `kwargs` are not used, but included for accordance with the Keras API
        x = tf.squeeze(inputs, axis=-1)
        fft = tf.signal.fft(tf.cast(x, tf.complex64))
        return tf.expand_dims(tf.abs(fft), axis=-1)


def create_generator(config: dict):
    inputs = layers.Input(shape=(config['noise_dim'],))
    
    x = layers.Dense(config['generator_initial_dense'], activation='relu')(inputs)
    x = layers.Reshape((config['generator_initial_dense'], 1))(x)
    for layer_config in config['generator_conv_layers']:
        x = layers.Conv1DTranspose(
            filters=layer_config['filters'],
            kernel_size=layer_config['kernel_size'],
            strides=layer_config['strides'],
            padding='same',
            activation='relu'
        )(x)
    assert x.shape[1] == config['signal_length'], f'Expected {config['signal_length']} but got {x.shape[1]}'
    
    outputs = layers.Conv1D(
        filters=1, kernel_size=config['generator_output_kernel'], padding='same', activation='tanh'
    )(x)
    return Model(inputs, outputs)


def create_critic(config: dict):
    inputs = layers.Input(shape=(config['signal_length'], 1))
    
    # Branch 1: Standard convolution
    branch1 = layers.Conv1D(
        config['critic_branch1']['filters'],
        kernel_size=config['critic_branch1']['kernel_size'],
        activation='relu',
        padding='same'
    )(inputs)
    branch1 = layers.MaxPooling1D(
        pool_size=config['critic_branch1']['pool_size']
    )(branch1)

    # Branch 2: FFT preprocessing
    branch2 = FFTPreprocessLayer()(inputs)
    branch2 = layers.Conv1D(
        config['critic_branch2']['filters'],
        kernel_size=config['critic_branch2']['kernel_size'],
        activation='relu',
        padding='same'
    )(branch2)
    branch2 = layers.MaxPooling1D(
        pool_size=config['critic_branch2']['pool_size']
    )(branch2)

    # Merge branches
    merged = layers.concatenate([branch1, branch2])

    # Final conv layer
    x = layers.Conv1D(
        config['critic_final_conv']['filters'],
        kernel_size=config['critic_final_conv']['kernel_size'],
        activation='relu',
        padding='same'
    )(merged)
    x = layers.GlobalAveragePooling1D()(x)

    outputs = layers.Dense(1)(x)
    return Model(inputs, outputs)


def gradient_penalty(critic, real_samples, fake_samples):
    batch_size = real_samples.shape[0]
    epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    return tf.reduce_mean((norm - 1.0) ** 2)


def train_wgan_gp(generator, critic, X_real, noise_dim=100, epochs=100, batch_size=32, n_critic=5, gp_weight=10):
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    log_dir = '../logs/wgan/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    summary_writer = tf.summary.create_file_writer(log_dir)

    for epoch in range(epochs):
        # 训练批判者
        for _ in range(n_critic):
            idx = np.random.randint(0, X_real.shape[0], batch_size)
            real_batch = X_real[idx]
            noise = np.random.randn(batch_size, noise_dim)
            fake_batch = generator(noise, training=False)

            with tf.GradientTape() as tape:
                real_output = critic(real_batch, training=True)
                fake_output = critic(fake_batch, training=True)
                c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gp = gradient_penalty(critic, real_batch, fake_batch)
                total_c_loss = c_loss + gp_weight * gp
            c_grads = tape.gradient(total_c_loss, critic.trainable_variables)
            c_optimizer.apply_gradients(zip(c_grads, critic.trainable_variables))

        # 训练生成器
        noise = np.random.randn(batch_size, noise_dim)
        with tf.GradientTape() as tape:
            fake_batch = generator(noise, training=True)
            fake_output = critic(fake_batch, training=False)
            g_loss = -tf.reduce_mean(fake_output)
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('Generator Loss', g_loss, step=epoch)
            tf.summary.scalar('Critic Loss', c_loss, step=epoch)
            tf.summary.scalar('Gradient Penalty', gp, step=epoch)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Critic Loss: {c_loss.numpy()}, GP: {gp.numpy()}, Generator Loss: {g_loss.numpy()}')
