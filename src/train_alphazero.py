from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from eurobot_env import EurobotDiscreteEnv
from alphazero import AlphaZeroConfig, AlphaZeroNetwork, AlphaZeroTrainer, EurobotActionHelper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero training for EurobotDiscreteEnv.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations.")
    parser.add_argument("--games-per-iter", type=int, default=4, help="Self-play games per iteration.")
    parser.add_argument("--num-simulations", type=int, default=128, help="MCTS simulations per move.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet noise alpha at root.")
    parser.add_argument("--dirichlet-eps", type=float, default=0.25, help="Dirichlet noise mixing weight.")
    parser.add_argument("--batch-size", type=int, default=128, help="Replay buffer batch size.")
    parser.add_argument("--training-steps", type=int, default=200, help="Gradient steps per iteration.")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--value-loss-coef", type=float, default=1.0, help="Value loss coefficient.")
    parser.add_argument("--policy-loss-coef", type=float, default=1.0, help="Policy loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=5.0, help="Gradient clipping norm.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Initial sampling temperature.")
    parser.add_argument("--temperature-decay-steps", type=int, default=30, help="Move index to anneal temperature.")
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256], help="MLP hidden layer sizes.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Environment RNG seed.")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to write periodic checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Iterations between checkpoints.")
    parser.add_argument("--save-model", type=str, help="Path to save final network weights.")
    parser.add_argument("--metrics-json", type=str, help="Optional path to dump training metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = EurobotDiscreteEnv(seed=args.seed)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = EurobotActionHelper(env).size()

    network = AlphaZeroNetwork(obs_dim, action_dim, hidden_sizes=args.hidden_sizes)

    cfg = AlphaZeroConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_eps,
        replay_buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        value_loss_coef=args.value_loss_coef,
        policy_loss_coef=args.policy_loss_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        temperature_decay_steps=args.temperature_decay_steps,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        checkpoint_every=args.checkpoint_every,
    )

    trainer = AlphaZeroTrainer(env, network, cfg)
    metrics = trainer.train()

    for entry in metrics:
        message = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(entry.items()))
        print(message)

    if args.save_model:
        path = Path(args.save_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.network.state_dict(), path)

    if args.metrics_json:
        path = Path(args.metrics_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
