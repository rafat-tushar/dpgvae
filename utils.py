import os
import numpy as np
import tensorflow as tf 
import gym 
from collections import deque
import imageio

class Logger:
    def __init__(self, exp_name = 'unknown', save_dir=''):
        self.results_dict = dict()
        self.config_dict = dict()
        self.eval_dict = dict()

        self.exp_name = exp_name 
        self.save_dir = save_dir

        if self.save_dir != '': os.makedirs(self.save_dir, exist_ok=True)
        self.file_prefix = os.path.join(self.save_dir, self.exp_name)

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not k in self.results_dict.keys():
                self.results_dict[k] = []
            self.results_dict[k].append(v)
    
    def store_config(self, **kwargs):
        """
        Store one time value like configurations parameters, variables.
        Variables with same name will replace previous values and update 
        with new values. So, this is suitable for one time, single(not list) 
        variables storing
        """
        config_data = ''
        for k, v in kwargs.items():
            # if not k in self.config_dict.keys():
            self.config_dict[k] = v 
            config_data += str(k) + ' : ' + str(v) + '\n'
        
        with open(self.file_prefix+'_config_info.txt', 'w') as f:
            f.write(config_data)
        self.results_dict['config'] = self.config_dict
        self.eval_dict['config'] = self.config_dict

    def store_eval(self, **kwargs):
        for k, v in kwargs.items():
            if not k in self.eval_dict.keys():
                self.eval_dict[k] = []
            self.eval_dict[k].append(v)
    
    def save(self):
        np.save(self.file_prefix+'_training_result.npy', self.results_dict)
        np.save(self.file_prefix+'_eval_result.npy', self.eval_dict)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.obs2_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.gamma_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, gamma):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.gamma_buf[self.ptr] = gamma 
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch_data = dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    gamma=self.gamma_buf[idxs])
        return {key: tf.convert_to_tensor(val, dtype='float32') for key,val in batch_data.items()}


class ActionNoise:
    def __init__(self, action_shape, init_noise=1.0, final_noise=0.1, noise_upto=100000):
        self.action_shape = action_shape
        self.init_noise = init_noise
        self.final_noise = final_noise
        self.noise_upto = noise_upto
        self.act_bound = 1.0 - 1e-6
        self.train_bound = 0.3
        self.curr_step = 1
    
    def get_action(self, synched_step=None):
        if synched_step is not None:
            self.curr_step = synched_step
        noise_temp = min(self.curr_step / self.noise_upto, 1.0)
        sigma = noise_temp * self.final_noise + (1.0 - noise_temp) * self.init_noise

        std = np.ones(self.action_shape) * sigma 
        gaussian_noise = np.random.normal(0.0, 1.0, self.action_shape)
        gaussian_noise *= std 
        if synched_step is None:
            gaussian_noise = np.clip(gaussian_noise, -self.train_bound, self.train_bound).astype(np.float32)
        return gaussian_noise


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        # self.env = env 
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=-1)


class VideoRecorder:
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        os.makedirs(self.dir_name, exist_ok=True)
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

def add_image_noise(img):
    img = img + tf.random.uniform(img.shape) / 32. 
    return img - .5 
