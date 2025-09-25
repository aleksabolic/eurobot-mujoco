# Eurobot RL 

Fast simulator for experimenting with Eurobot match tactics. The repo provides a discrete Gym environment, PPO training loop, scripted opponent policy, and utilities for playback/rendering.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
Launch PPO training (Stable-Baselines3) from the repo root:

```bash
python src/train.py \
--timesteps 100_000_000 \
--ckpt runs/<checkpoint-name>
```

TensorBoard logs are written under `runs/tb/<checkpoint-name>`

## Evaluation & Rendering
```bash
python src/eval.py --model runs/<checkpoint-name>.zip --render --fps 5
```
Add `--render` for live OpenCV playback, adjust `--fps` for slower/faster replays, or `--gif path/to/output.gif` to export animations (requires `imageio`).

## Robot Configuration
Robot motion/handling profiles and opponent heuristics live in `robot_configs/blue_robot.json` and `robot_configs/yellow_robot.json`.

## Project Layout
- `src/eurobot_env.py` – Gymnasium wrapper around the Eurobot world with discrete actions.
- `src/eurobot_world.py` – Core match simulation, scoring, and logistics logic.
- `src/masked_policy.py` & `src/policies.py` – Custom SB3 policy with action masking and scripted opponent.
- `src/train.py`, `src/eval.py` – PPO training loop and evaluation utility.
- `assets/` – Arena layout and geometry assets used by the simulator.
- `robot_configs/` – Editable robot capability/policy presets.
- `tests/` – PyTest suite covering world mechanics edge cases.

Run `pytest` from the repo root to validate environment dynamics after changing core logic or configs.
