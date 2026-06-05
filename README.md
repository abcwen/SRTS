# SRTS

This project implements the reproducible core method described in **SRTS: Semantic Representations for Trajectory Stitching in Offline Reinforcement Learning**.

The implementation follows the paper pipeline:

1. **Semantic representation learning** with a Goal-Conditioned Trajectory Semantic Model (GCTSM).
2. **Hierarchical compatibility selection** using slot-pooled semantic cosine similarity plus normalized boundary-state distance.
3. **Bridge generation** with a conditional DDPM trajectory stitcher.
4. **Transition reconstruction** with inverse dynamics, reward, and forward dynamics consistency filtering.

## Installation

Create an environment:

```bash
pip install -r requirements.txt
pip install -e .
```

D4RL is optional. If D4RL is unavailable on your platform, pass a local `.npz` dataset with the following keys:

```text
observations, actions, rewards, next_observations, terminals, episode_ends
```

`terminals` contains true MDP terminal flags for offline RL bootstrapping. `episode_ends` additionally includes time-limit boundaries and is used only for trajectory segmentation. 

## Run

For example:

```bash
python scripts/run_srts.py --config configs/configs.yaml data.dataset=halfcheetah-medium-v2
```

Each run now writes to a short fingerprinted experiment directory:

```text
runs/halfcheetah-medium-v2/
  run_a1b2c3d4e5f6/
    augmented_srts.npz
    metrics.json
    manifest.json
    checkpoints/
```

`manifest.json` records the complete configuration, full dataset-content hash, model dimensions, runtime selection options, implementation version, and hashes of relevant source files.
