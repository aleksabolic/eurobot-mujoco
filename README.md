# Eurobot MuJoCo Starter

Minimal, fast simulator to explore Eurobot 2026 tactics with RL.
- 2D-ish MuJoCo world (3x2 m), two differential-drive robots, crates, pantries.
- PettingZoo ParallelEnv (`EurobotMJ`) with entity-centric observations.
- PPO self-play scaffold: train vs scripted bot, then load last checkpoint as opponent.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_selfplay.py --timesteps 2_000_000
```
Evaluate:
```bash
python eval.py --model runs/ppo_blue_last.zip
```

## Files
- `assets/`: MuJoCo MJCF arena + bodies.
- `eurobot_env.py`: PettingZoo env with simple pick/drop logic.
- `train_selfplay.py`: PPO training (SB3), opponent can be scripted or last checkpoint.
- `eval.py`: deterministic rollout and print score.
