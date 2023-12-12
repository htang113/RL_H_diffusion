Reinforcement learning guided long timescale simulation

Folder rlmd: Classes for neural network model, RL training, interatomic potential and KMC environment, and trajectory loader
Folder POSCARs: Initial structure files
train_DQN.py, train_context_bandit.py: code to launch deep Q network in H diffusion in Cu slab or contextual bandit training in H diffusion in CrCoNi medium entropy alloy
deploy_DQN.py, deploy_context_bandit.py: code to deploy the deep Q network or contextual bandit model and sample trajectories
model_DQN.pt, model_context_bandit.py: trained model files for deploy_DQN.py and deploy_context_bandit.py to load and deploy.

