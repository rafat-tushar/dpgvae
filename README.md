# DPG-VAE

This repository presents the implementation of

**Sample Efficient Reinforcement Learning from Visual Data with Modified Deep Deterministic Policy Gradient and Variational Autoencoder**

by [Rafat Rahman Tushar]() and [Shahnewaz Siddique]()

|[Paper]()| |[Video]()| 

## Results
Our DPG-VAE is evaluated on visual data received from various continuous control tasks of the DM-Control suite. We compared our evaluation results with various computationally intensive model-based and model-free methods. We represent our state-based low-dimensional model as DPG-STATE. DPG-STATE acts as an Oracle so that the upper bound for each task can be known. The evaluation results illustrate that DPG-VAE outperforms other state-of-the-art models on most of the tasks at various training steps.

<p float="left">
  <img src="figures/ball_cup_catch_eval_result.png" width="32%" />
  <img src="figures/cartpole_swingup_eval_result.png" width="32%" />
  <img src="figures/cheetah_run_eval_result.png" width="32%" />
  <img src="figures/finger_spin_eval_result.png" width="32%" />
  <img src="figures/reacher_easy_eval_result.png" width="32%" />
  <img src="figures/walker_walk_eval_result.png" width="32%" />
</p>
