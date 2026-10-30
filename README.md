# FaStR

FaStR is a reinforcement learning method that learns separate features for
states and actions. This repository contains its code and training settings.


## Install

Use Python 3.10 or later. Create a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the packages for the tasks you want to run:

```bash
# DeepMind Control Suite
pip install -e '.[dmc]'

# Gymnasium MuJoCo
pip install -e '.[mujoco]'

# Adroit and Shadow Hand
pip install -e '.[robotics]'

# All task groups
pip install -e '.[dmc,mujoco,robotics]'
```

The task packages use MuJoCo 3.2.7. The old `mujoco-py` package is not needed.
PyTorch uses CUDA when it is available, or CPU otherwise. Set `device=cpu`,
`device=0`, or `device=cuda:1` to choose a device.

## Train

All tasks use the same training command:

```bash
python -m fastr.train env=dmc task=hopper-hop seed=0
python -m fastr.train env=mujoco task=HalfCheetah-v5 seed=0
python -m fastr.train env=adroit task=AdroitHandDoor-v1 seed=0
python -m fastr.train env=robotics task=HandReach-v2 seed=0
```

After installation, `fastr-train` runs the same command. Task names depend on
the installed task package version.

The `adroit` settings use hidden layers with 512 units and a hybrid critic.
The `robotics` settings use hidden layers with 256 units and a bilinear critic.
A bilinear critic combines state and action features through a weight matrix.
A hybrid critic adds a neural network term to that value.

Change settings on the command line:

```bash
python -m fastr.train env=dmc task=walker-walk \
  train_frames=1000000 batch_size=512 algo.d_s=16 algo.d_a=16

# Use a diagonal transition model
python -m fastr.train algo.diagonal_m=true algo.d_s=512 algo.d_a=512

# Add an exploration bonus to the actor loss
python -m fastr.train algo.bonus_coef=0.02 algo.thompson_sampling=false

# Show the full settings
python -m fastr.train --cfg job
```

The default exploration bonus is zero. All task settings use a full bilinear
transition model by default. The diagonal option uses fewer model outputs.

`algo.critic_type` can be `bilinear`, `hybrid`, `product_kernel`,
`spectral_residual`, or `rff_psi`. The last two need the same number of state
and action features.

`algo.bonus_type` can be `factored`, `action_only`, or `joint`. These options
use separate state and action features, action features only, or their
product to measure uncertainty. The `joint` option also needs the same
number of state and action features.

## Logs

Each run saves its settings and TensorBoard logs to:

```text
log/fastr/<name>/<task>/<run>/
```

View the logs with:

```bash
tensorboard --logdir log/fastr
```

Set `debug=true` to turn off file logging. Training starts after the replay
buffer holds at least `batch_size` samples and `warmup_frames` have passed.
When an episode ends at a time limit, training still uses the next state's
value. When the task ends, that future value is zero.

## Code source and license

This code is based on [Spectral-RL](https://github.com/spectral-rl/spectral-rl) with modifications.
We thank the original authors for sharing their code under the MIT License. The code uses the same MIT License.

## Files

```text
fastr/
├── agent.py       # FaStR updates and action selection
├── network.py     # State and action encoders and model loss
├── critic.py      # Value networks
├── nn.py          # Shared network layers
├── replay.py      # Replay buffer and running data statistics
├── train.py       # Training and evaluation
├── envs/          # Task wrappers
├── config/        # Shared settings and task settings
├── logger.py      # TensorBoard logs
└── utils.py       # Seeds, devices, and target network updates
```

## Citation

If you find our work useful, please consider citing our NeurIPS 2026 paper:

```bibtex
@inproceedings{wu2026fastr,
  title     = {Factorized Spectral Representations for Reinforcement Learning},
  author    = {Wu, Junyi and Li, Dan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://neurips.cc/virtual/2026/loc/atlanta/poster/149364}
}
```

