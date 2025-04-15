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
    x = inputs
    x = _build_network(x, config['generator_layers'])
    outputs = x
    return Model(inputs, outputs)


def _build_network(input_tensor: tf.Tensor, layer_configs: list):
    x = input_tensor
    for layer_config in layer_configs:
        layer_name = layer_config['name']
        if layer_name == 'Conv1D':
            x = layers.Conv1D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                strides=layer_config.get('strides', 1),
                padding=layer_config.get('padding', 'same'),
                activation=layer_config.get('activation', 'relu')
            )(x)
        elif layer_name == 'Dropout':
            x = layers.Dropout(rate=layer_config['rate'])(x)
        elif layer_name == 'MaxPooling1D':
            x = layers.MaxPooling1D(
                pool_size=layer_config['pool_size'],
                strides=layer_config['strides']
            )(x)
        elif layer_name == 'GlobalAveragePooling1D':
            x = layers.GlobalAveragePooling1D()(x)
        elif layer_name == 'Dense':
            x = layers.Dense(
                units=layer_config['units'],
                activation=layer_config.get('activation', 'relu')
            )(x)
        elif layer_name == 'BatchNormalization':
            x = layers.BatchNormalization()(x)
        elif layer_name == 'Reshape':
            x = layers.Reshape(target_shape=layer_config['target_shape'])(x)
        elif layer_name == 'Conv1DTranspose':
            x = layers.Conv1DTranspose(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                strides=layer_config.get('strides', 1),
                padding=layer_config.get('padding', 'same'),
                activation=layer_config.get('activation', 'relu')
            )(x)
    return x


def create_critic(config: dict):
    inputs = layers.Input(shape=(config['signal_length'], 1))
    
    # Branch 1: Standard convolution
    branch1 = inputs
    branch1 = _build_network(branch1, config['critic_branch1_layers'])

    # Branch 2: FFT preprocessing
    branch2 = FFTPreprocessLayer()(inputs)
    branch2 = _build_network(branch2, config['critic_branch2_layers'])

    # Merge branches
    merged = layers.concatenate([branch1, branch2])
    merged = _build_network(merged, config['critic_merged_layers'])

    outputs = layers.Dense(1)(merged)
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
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00008, beta_1=0.5, beta_2=0.9)
    c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00004, beta_1=0.5, beta_2=0.9)

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
