import tensorflow as tf 
from enc_dec import Encoder

class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, latent_dim):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(obs_dim, latent_dim)

        self.base = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(self.latent_dim))])
        for hidden_size in self.hidden_sizes:
            self.base.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
        self.base.add(tf.keras.layers.Dense(self.act_dim, activation=None))
    
    def call(self, obs):
        latent_obs = self.encoder.encode(obs)
        mu = self.base(latent_obs)
        return tf.math.tanh(mu) 


class Critic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, latent_dim):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.latent_dim = latent_dim

        self.encoder = Encoder(obs_dim, latent_dim)

        self.q1 = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(self.latent_dim + self.act_dim))])
        for hidden_size in self.hidden_sizes:
            self.q1.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
        self.q1.add(tf.keras.layers.Dense(1, activation=None))

        self.q2 = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(self.latent_dim + self.act_dim))])
        for hidden_size in self.hidden_sizes:
            self.q2.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
        self.q2.add(tf.keras.layers.Dense(1, activation=None))

    def call(self, obs, act):
        latent_obs = self.encoder.encode(obs)
        obs_act = tf.concat([latent_obs, act], axis=-1)
        
        Q1 = self.q1(obs_act)
        Q2 = self.q2(obs_act)
        return tf.squeeze(Q1, axis=-1), tf.squeeze(Q2, axis=-1)
