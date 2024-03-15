import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(*self.obs_dim,)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,  activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,  activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,  activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
        ])
    
    def encode(self, obs):
        mean, _ = tf.split(self.encoder(obs), num_or_size_splits=2, axis=1)
        return mean 

    def reparameterize(self, obs):
        mean, log_std = tf.split(self.encoder(obs), num_or_size_splits=2, axis=1)
        eps = tf.random.normal(mean.shape, 0.0, 1.0)
        return mean, log_std, eps * tf.exp(log_std * .5) + mean


class Decoder(tf.keras.Model):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim 

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
            tf.keras.layers.Dense(units=34*34*32, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(34, 34, 32)),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1,activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1,activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1,activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1,activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=9, kernel_size=3, strides=2, padding='same'),
        ])
    
    def decode(self, z):
        logits = self.decoder(z)
        return logits
