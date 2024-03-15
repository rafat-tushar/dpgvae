from collections import deque
import sys 
import time 
from datetime import datetime
import copy 
import random
import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt 
import os 
import gym 

from core import Actor, Critic
from enc_dec import Decoder
from utils import Logger, ReplayBuffer, ActionNoise, FrameStack


class DPGAE(tf.keras.Model):
    def __init__(self, config):
        super(DPGAE, self).__init__()
        self.config = config
        experiment_dir = f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
        self.logger = Logger(exp_name=config['model'], save_dir=experiment_dir)
        self.encoder_image_path = experiment_dir + '/encoder_figs/'
        os.makedirs(self.encoder_image_path, exist_ok=True)

        self.max_ep_len = config['max_ep_length']
        self.save_freq = config['save_frequency']
        self.num_eval_episodes = config['num_eval_episodes']
        self.batch_size = config['batch_size']
        self.num_train_steps = config['number_of_training_steps']
        self.steps_per_epoch = config['steps_per_epoch'] 
        self.n_steps_return = config['n_steps_return']
        self.primary_random_steps = config['primary_random_steps'] 
        self.update_every = config['gradient_update_every'] 
        self.seed = config['random_seed']

        self.noise_init = config['initial_noise_value']
        self.noise_final = config['final_noise_value']
        self.noise_steps_upto = config['noise_upto_steps']

        self.frame_skip = config['frame_skip']
        self.frame_stack = config['frame_stack']

        # Setting seed everywhere 
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Simulation environment setup
        env = gym.make(config['env'], frame_skip=self.frame_skip, from_pixels=True, channels_first=False)
        eval_env = gym.make(config['env'], frame_skip=self.frame_skip, from_pixels=True, channels_first=False)
        self.env = FrameStack(env, self.frame_stack)
        self.eval_env = FrameStack(eval_env, self.frame_stack)

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high[0] - 1e-6
        self.action_low = self.env.action_space.low[0]

        # Gaussian Noise generator
        self.gauss_noise = ActionNoise([self.batch_size, self.act_dim], self.noise_init, self.noise_final, self.noise_steps_upto)
        self.latent_lambda = 1e-6 
        
        self.hidden_sizes = config['hidden_sizes']
        self.encoder_latent_dim = config['latent_dim']
        self.actor_lr = config['actor_learning_rate']
        self.critic_lr = config['critic_learning_rate']
        self.encoder_lr = config['encoder_learning_rate']
        self.tau_critic = config['critic_soft_update_tau']
        self.tau_encoder = config['encoder_soft_update_tau']
        self.gamma = config['discount_rate']

        # Model 
        self.actor = Actor(self.obs_dim, self.act_dim, self.hidden_sizes, self.encoder_latent_dim)
        self.critic = Critic(self.obs_dim, self.act_dim, self.hidden_sizes, self.encoder_latent_dim)
        self.critic_targ = Critic(self.obs_dim, self.act_dim, self.hidden_sizes, self.encoder_latent_dim)
        self.decoder = Decoder(self.obs_dim, self.encoder_latent_dim)

        self.critic_targ.set_weights(self.critic.get_weights()) 
        self.copy_weights(self.critic.encoder, self.actor.encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate = self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate = self.critic_lr)
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate = self.encoder_lr)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, config['replay_buffer_size'])

    def compute_critic_loss(self, obs, act, rew, obs_nth, gamma, added_noise):
        q1, q2 = self.critic(obs, act)
        
        mu_nth = self.actor(obs_nth)
        act_nth = self.action_clip(mu_nth + added_noise)
        q1_nth, q2_nth = self.critic_targ(obs_nth, act_nth)

        q_nth = tf.minimum(q1_nth, q2_nth)
        backup = tf.stop_gradient(rew + gamma * q_nth)

        critic_loss = 0.5 * tf.reduce_mean((backup-q1)**2) + 0.5 * tf.reduce_mean((backup-q2)**2)

        return critic_loss, q1, q2, backup 
    
    def compute_actor_loss(self, obs, added_noise):
        mu = self.actor(obs)
        act = self.action_clip(mu + added_noise)

        q1, q2 = self.critic(obs, act)
        q = tf.minimum(q1, q2)

        actor_loss = -tf.reduce_mean(q)

        return actor_loss
    
    
    def log_normal_pdf(self, sample, mean, logstd, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logstd) + logstd + log2pi),axis=raxis)


    def compute_encoder_loss(self, obs):
        mean, logstd, z = self.critic.encoder.reparameterize(obs)
        x_logit = self.decoder.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=obs)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logstd)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    @tf.function
    def update_step(self, data, ac_noise, cr_noise):
        main_actor_vars = self.actor.base.trainable_variables
        main_critic_vars = self.critic.q1.trainable_variables + self.critic.q2.trainable_variables
        main_autoencoder_vars = self.critic.encoder.trainable_variables + self.decoder.trainable_variables

        obs, obs_n, act, rew, gamma = data['obs1'],data['obs2'],data['acts'],data['rews'],data['gamma'] 
        obs = obs / 255.0
        obs_n = obs_n / 255.0

        # AutoEncoder update 
        with tf.GradientTape() as tape:
            encoder_loss = self.compute_encoder_loss(obs[:128])
        e_gradients = tape.gradient(encoder_loss, main_autoencoder_vars)
        self.encoder_optimizer.apply_gradients(zip(e_gradients, main_autoencoder_vars))

        # Critic update
        with tf.GradientTape() as tape:
           critic_loss, q1, q2, y = self.compute_critic_loss(obs, act, rew, obs_n, gamma, cr_noise)
        c_gradients = tape.gradient(critic_loss, main_critic_vars)
        self.critic_optimizer.apply_gradients(zip(c_gradients, main_critic_vars))

        # Actor update
        with tf.GradientTape() as tape:
            actor_loss = self.compute_actor_loss(obs, ac_noise)
        a_gradients = tape.gradient(actor_loss, main_actor_vars)
        self.actor_optimizer.apply_gradients(zip(a_gradients, main_actor_vars))

        return actor_loss, critic_loss, encoder_loss, tf.reduce_mean(rew), q1, q2, y
    
    def update(self, steps):
        sampled_data = self.replay_buffer.sample_batch(self.batch_size)
        actor_noise = self.gauss_noise.get_action()
        critic_noise = self.gauss_noise.get_action()
        
        update_time = time.time()
        
        actor_loss, critic_loss, encoder_loss, _, _, _, _ = self.update_step(sampled_data, actor_noise, critic_noise)

        if steps % 2 == 0:
            # Target weights update
            self.target_weights_update(self.critic.q1, self.critic_targ.q1, self.tau_critic)
            self.target_weights_update(self.critic.q2, self.critic_targ.q2, self.tau_critic)
        
        self.target_weights_update(self.critic.encoder, self.critic_targ.encoder, self.tau_encoder)

        if steps % 50 == 0:
            # Encoder sync
            self.copy_weights(self.critic.encoder, self.actor.encoder)
        
        self.logger.store(
            ActorLoss=actor_loss.numpy(), CriticLoss=critic_loss.numpy(),
            EncoderLoss=encoder_loss.numpy(), UpdateTime=time.time() - update_time
        )
        
        return actor_loss, critic_loss, encoder_loss


    @tf.function
    def target_weights_update(self, source, target, tau):
        # Soft update of target network from source 
        for source_var, target_var in zip(source.variables, target.variables):
            target_var.assign(target_var * (1.0 - tau) + source_var * tau)
    
    @tf.function
    def copy_weights(self, source, target):
        # Copy weights from source to target network 
        for source_var, target_var in zip(source.variables, target.variables):
            target_var.assign(source_var)

    
    def get_action(self, obs, steps, eval_mode=False):
        obs = obs.astype('float32') / 255.0 
        o = tf.expand_dims(obs, 0)
        act = self.actor(o)
        if not eval_mode:
            noise = self.gauss_noise.get_action(steps)
            noise = np.expand_dims(noise[0], 0)
            act = self.action_clip(act + noise)
        return act.numpy()[0]
    
    def action_clip(self, action):
        return tf.clip_by_value(action, -self.action_high, self.action_high)
    
    def evaluate_agent(self, steps):
        eval_res = dict(TestEpRet=[], TestEpLen=[])
        for ep_count in range(self.num_eval_episodes):
            o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
            
            while not(d or (ep_len == self.max_ep_len)):
                # Get action in 'eval_mood' 
                a = self.get_action(o, None, True)
                o, r, d, _ = self.eval_env.step(a)
                ep_ret += r
                ep_len += 1
            
            eval_res['TestEpRet'].append(ep_ret)
            eval_res['TestEpLen'].append(ep_len)
            self.logger.store_eval(TestEpRet=ep_ret, TestEpLen=ep_len)
        print("---------------------------------------\n")
        for i, _ in enumerate(eval_res['TestEpRet']):
            print(f"Return: {eval_res['TestEpRet'][i]:4.2f}\t Length: {eval_res['TestEpLen'][i]:4.2f} ")
        print(
            f"Evaluation Summary (mean+-std): {np.mean(eval_res['TestEpRet']):5.4f}",
            f"+- {np.std(eval_res['TestEpRet']):5.4f}"
        )
        print("---------------------------------------")
        self.logger.store_eval(
            EvalRets=eval_res['TestEpRet'], EvalLens=eval_res['TestEpLen'],
            EvalMeanRets = np.mean(eval_res['TestEpRet']), Steps=steps,
            EvalStdRets = np.std(eval_res['TestEpRet'])
        )
    
    def train(self):
        start_time = time.time()
        epoch_start_time = time.time()

        self.exp_buffer = deque()

        o, ep_ret, ep_len = self.env.reset(), 0, 0
        self.exp_buffer.clear()
        epoch_result = dict(ep_rets=[], ep_lens=[])

        for t in range(self.num_train_steps):
            if t < self.primary_random_steps:
                a = self.env.action_space.sample()
            else:
                a = self.get_action(o, t)
            
            o2, r, d, _ = self.env.step(a)
            ep_ret += r 
            ep_len += 1 

            d = False if ep_len==self.max_ep_len else d 

            self.exp_buffer.append((o,a,r))

            if len(self.exp_buffer) >= self.n_steps_return:
                _o, _a, _r = self.exp_buffer.popleft()
                n_reward = _r 
                gamma = self.gamma 
                for (_, _, r_i) in self.exp_buffer:
                    n_reward += r_i * gamma 
                    gamma *= self.gamma
                self.replay_buffer.store(_o, _a, n_reward, o2, d, gamma)
            
            o = o2 

            # End of trajectory for n-step 
            if d or (ep_len == self.max_ep_len):
                while len(self.exp_buffer) != 0:
                    _o, _a, _r = self.exp_buffer.popleft()
                    n_reward = _r 
                    gamma = self.gamma 
                    for (_, _, r_i) in self.exp_buffer:
                        n_reward += r_i * gamma 
                        gamma *= self.gamma
                    self.replay_buffer.store(_o, _a, n_reward, o2, d, gamma)
                
                self.logger.store(EpRets=ep_ret, EpLens=ep_len)
                epoch_result['ep_rets'].append(ep_ret) 
                epoch_result['ep_lens'].append(ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                self.exp_buffer.clear()
            
            if t > self.batch_size:
                actor_loss, critic_loss, encoder_loss = self.update(t)
            
            # End of epoch wrap-up
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch 
                print(
                    f"Epoch:{epoch:3d} ALoss:{actor_loss:2.4f} CLoss:{critic_loss:2.4f} AELoss:{encoder_loss:2.4f} ",
                    f"EpRet:{np.mean(epoch_result['ep_rets']):2.4f} EpLen:{np.mean(epoch_result['ep_lens']):2.2f}",
                    f"TIME:{(time.time()-epoch_start_time)/60:2.2f} min"
                )

                self.logger.store(
                    BatchRets=np.mean(epoch_result['ep_rets']),
                    BatchLens=np.mean(epoch_result['ep_lens'])
                )
                epoch_result = dict(ep_rets=[], ep_lens=[])
                self.evaluate_agent(t+1)
                self.logger.save()
                epoch_start_time = time.time()
        self.logger.store(TrainingTime=time.time() - start_time)
        self.logger.save()
        print(f'\n Total running time: {self.logger.results_dict["TrainingTime"][0]/60:.3f} minutes \t Seed: {self.seed}')
