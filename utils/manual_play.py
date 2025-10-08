#!/usr/bin/env python3
"""
Interactive CLI to play a single Eurobot match by hand.
Pick actions for the BLUE robot and watch the scripted opponent respond.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import imageio
import numpy as np

# Make project sources importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eurobot_env import EurobotDiscreteEnv
from eurobot_world import Col, EurobotWorld, NodeType
from robot import Verb
from eurobot_render import EurobotCV2Renderer


@dataclass
class ParsedAction:
    verb: int
    node: int
    color: int
    qty: int


def build_node_lookup(world: EurobotWorld) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, node in enumerate(world.nodes):
        name = node.name.lower()
        lookup[name] = idx
        lookup[str(idx)] = idx
        lookup[name.replace(" ", "")] = idx
    return lookup


def format_inv(inv: np.ndarray) -> str:
    return f"blue={int(inv[Col.BLUE])}, yellow={int(inv[Col.YELLOW])}"


def print_state(world: EurobotWorld) -> None:
    blue_node = world.nodes[world.blue.node].name
    yellow_node = world.nodes[world.yellow.node].name
    print("\n=== Current State ===")
    print(f"Time left: {world.t_left:5.2f}s")
    print(f"Score  → blue={world.nest_blue_counted}, yellow={world.nest_yellow_counted}")
    print(f"Blue    → node={blue_node:>10}, inv=({format_inv(world.blue.inv)})")
    print(f"Yellow  → node={yellow_node:>10}, inv=({format_inv(world.yellow.inv)})")

    def _print_stock(title: str, entries: Iterable[Tuple[int, np.ndarray]]) -> None:
        lines = []
        for node_id, stock in entries:
            if int(stock.sum()) == 0:
                continue
            lines.append(
                f"  {node_id:2d} {world.nodes[node_id].name:<10}  "
                f"B={int(stock[Col.BLUE])}  Y={int(stock[Col.YELLOW])}"
            )
        print(title)
        if lines:
            for line in lines:
                print(line)
        else:
            print("  (empty)")

    pantry_entries = ((node_id, world.pantries[j]) for j, node_id in enumerate(world.PANTRIES))
    pickup_entries = ((node_id, world.pickups[j]) for j, node_id in enumerate(world.PICKUPS))
    _print_stock("Pantries with stock:", pantry_entries)
    _print_stock("Pickups with stock:", pickup_entries)

    if world.last_invalid_detail:
        detail = world.last_invalid_detail
        print("\nLast invalid action detail:")
        for key, value in detail.items():
            print(f"  {key}: {value}")


def print_nodes(world: EurobotWorld) -> None:
    print("\nAvailable nodes:")
    for idx, node in enumerate(world.nodes):
        kind = NodeType(node.kind).name
        xy = ", ".join(f"{coord:+.2f}" for coord in node.xy)
        print(f"  {idx:2d}: {node.name:<10} [{kind}] at ({xy})")


def render_and_print(world: EurobotWorld, renderer: Optional[EurobotCV2Renderer], show_window: bool) -> None:
    if renderer is not None:
        renderer.draw_snapshot(show=show_window)
    print_state(world)


VERB_LOOKUP = {v.name.lower(): int(v) for v in Verb}
for v in Verb:
    VERB_LOOKUP[str(int(v))] = int(v)

COLOR_LOOKUP = {
    "blue": int(Col.BLUE),
    "b": int(Col.BLUE),
    "0": int(Col.BLUE),
    "yellow": int(Col.YELLOW),
    "y": int(Col.YELLOW),
    "1": int(Col.YELLOW),
}


def parse_action(
    raw: str,
    world: EurobotWorld,
    actor_tag: str,
    node_lookup: Dict[str, int],
) -> ParsedAction:
    tokens = raw.strip().split()
    if not tokens:
        raise ValueError("no input provided")

    if tokens[0].lower() not in VERB_LOOKUP:
        raise ValueError(f"unknown verb '{tokens[0]}'")
    verb = VERB_LOOKUP[tokens[0].lower()]

    current_node = world.blue.node if actor_tag == "blue" else world.yellow.node
    node = current_node
    color = int(Col.BLUE)
    qty_defaults = {
        Verb.FLIP: 1,
        Verb.PICK: 1,
        Verb.PLACE: 1,
        Verb.STEAL: 1,
    }
    qty = qty_defaults.get(Verb(verb), 0)

    idx = 1
    if Verb(verb) in (Verb.PICK, Verb.PLACE, Verb.STEAL):
        if idx >= len(tokens):
            node = current_node
        else:
            node_token = tokens[idx].lower()
            idx += 1
            if node_token != "-":
                if node_token not in node_lookup:
                    raise ValueError(f"unknown node '{tokens[idx-1]}'")
                node = node_lookup[node_token]

    if Verb(verb) in (Verb.PICK, Verb.PLACE, Verb.FLIP, Verb.STEAL):
        if idx < len(tokens):
            color_token = tokens[idx].lower()
            idx += 1
            if color_token != "-":
                if color_token not in COLOR_LOOKUP:
                    raise ValueError(f"unknown color '{tokens[idx-1]}'")
                color = COLOR_LOOKUP[color_token]
        else:
            color = int(Col.BLUE)

    if idx < len(tokens):
        qty_token = tokens[idx]
        if qty_token != "-":
            qty = int(qty_token)

    return ParsedAction(verb=verb, node=node, color=color, qty=qty)


def prompt_action(world: EurobotWorld, actor_tag: str, node_lookup: Dict[str, int]) -> ParsedAction:
    print_nodes_hint = True
    while True:
        prompt = f"{actor_tag.upper()} action [verb node color qty] (type 'help'): "
        raw = input(prompt)
        lowered = raw.strip().lower()
        if lowered in {"help", "h", "?"}:
            print("Enter actions like 'pick P1 blue 2' or 'place pantryA blue 2'. "
                  "Use '-' to keep defaults (current node/color/qty).")
            if print_nodes_hint:
                print("Type 'nodes' to list all node names.")
                print_nodes_hint = False
            continue
        if lowered in {"nodes", "n"}:
            print_nodes(world)
            continue
        if lowered in {"state", "status", "s"}:
            print_state(world)
            continue
        if lowered in {"quit", "exit"}:
            raise SystemExit("User aborted the match.")

        try:
            action = parse_action(raw, world, actor_tag, node_lookup)
            return action
        except ValueError as exc:
            print(f"Could not parse action: {exc}")


def execute_user_action(
    env: EurobotDiscreteEnv,
    blue_action: ParsedAction,
    renderer: Optional[EurobotCV2Renderer],
    show_window: bool,
    total_steps: int,
) -> tuple[bool, Dict[str, float], float, int]:
    cumulative_reward = 0.0
    steps_taken = 0
    info: Dict[str, float] = {}
    done = False

    def run_step(action_arr: np.ndarray, *, auto: bool) -> None:
        nonlocal cumulative_reward, steps_taken, info, done, total_steps
        label = "Auto-advance" if auto else "Action"
        obs, reward, terminated, truncated, step_info = env.step(action_arr)
        _ = obs  # observation not needed for CLI, but keep for clarity
        total_steps += 1
        steps_taken += 1
        cumulative_reward += reward
        print(f"\n--- {label} step {total_steps} ---")
        print(f"Reward: {reward: .3f}")
        if step_info:
            for key, value in step_info.items():
                print(f"{key}: {value}")
        render_and_print(env.world, renderer, show_window)
        info = step_info
        done = bool(terminated or truncated)

    action_arr = np.array(
        [blue_action.verb, blue_action.node, blue_action.color, blue_action.qty],
        dtype=np.int64,
    )
    run_step(action_arr, auto=False)

    while not done and env.world.blue.event is not None and env.world.t_left > 0.0:
        dt, r_gain = env.world._advance_until_next()
        if dt <= 0.0:
            break
        time_penalty = env.world.rewards.time_penalty * dt
        auto_reward = r_gain
        if abs(time_penalty) > 1e-9:
            auto_reward -= time_penalty
            env.world._add_blue_return(-time_penalty)

        total_steps += 1
        steps_taken += 1
        cumulative_reward += auto_reward

        print(f"\n--- Auto-advance step {total_steps} ---")
        print(f"Reward: {auto_reward: .3f}")
        render_and_print(env.world, renderer, show_window)

        if env.world.t_left <= 1e-9:
            done = True
            bonus = env.world._terminal_bonus()
            if abs(bonus) > 1e-9:
                cumulative_reward += bonus
                env.world._add_blue_return(bonus)
            env.world._snap("end")
            info = {
                "blue_final_score": float(env.world.final_scores()[0]),
                "yellow_final_score": float(env.world.final_scores()[1]),
            }

    print(
        f"Action resolved in {steps_taken} internal step(s); "
        f"cumulative reward {cumulative_reward: .3f}."
    )
    return done, info, cumulative_reward, total_steps


def save_replay_gif(
    world: EurobotWorld,
    renderer: EurobotCV2Renderer,
    target_dir: Path,
    *,
    fps: int = 4,
) -> Optional[Path]:
    if not world.history:
        return None
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    frames = []
    history_len = len(world.history)
    for idx in range(history_len):
        frame_bgr = renderer.draw_snapshot(idx=idx, show=False)
        if frame_bgr is None:
            continue
        frame_rgb = frame_bgr[:, :, ::-1]
        frames.append(frame_rgb)

    if not frames:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = target_dir / f"manual_play_{timestamp}.gif"
    try:
        imageio.mimsave(gif_path, frames, fps=fps)
    except Exception:
        return None
    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play a single Eurobot match manually through the console.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment.")
    parser.set_defaults(render=True)
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable OpenCV board rendering (console output only).",
    )
    args = parser.parse_args()

    env = EurobotDiscreteEnv(seed=args.seed)
    world = env.world
    node_lookup = build_node_lookup(world)

    _obs, _ = env.reset(seed=args.seed)
    renderer = EurobotCV2Renderer(world)
    show_window = bool(args.render)

    print_nodes(world)
    render_and_print(world, renderer, show_window)
    print("\nControls:")
    print("  - Provide actions as: verb node color qty")
    print("  - Optional tokens can be '-' to accept defaults.")
    print("  - Commands: 'nodes', 'status', 'help', 'quit'")
    print("Starting match... good luck!\n")

    info: Dict[str, float] = {}
    done = False
    total_steps = 0
    match_completed = False
    gif_path: Optional[Path] = None
    try:
        while not done:
            world.last_invalid_detail = None

            blue_action = prompt_action(world, "blue", node_lookup)

            done, info, _, total_steps = execute_user_action(
                env,
                blue_action,
                renderer,
                show_window,
                total_steps,
            )
        match_completed = bool(done)
    finally:
        if renderer is not None and match_completed:
            try:
                gif_path = save_replay_gif(world, renderer, REPO_ROOT / "utils")
            except Exception as exc:
                print(f"\nFailed to save replay GIF: {exc}")
        if show_window:
            try:
                import cv2

                cv2.waitKey(1)
                cv2.destroyAllWindows()
            except Exception:
                pass

    blue_score = info.get("blue_final_score") if info else None
    yellow_score = info.get("yellow_final_score") if info else None
    print("\nMatch finished!")
    if blue_score is not None and yellow_score is not None:
        print(f"Final score → blue={blue_score}, yellow={yellow_score}")
    if gif_path is not None:
        print(f"Replay available at: {gif_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
