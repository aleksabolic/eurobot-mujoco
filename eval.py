import argparse, numpy as np
from stable_baselines3 import PPO
from eurobot_env import EurobotMJ

def run(model_path, episodes=5):
    env = EurobotMJ(xml_path="assets/arena.xml", max_steps=1200, scripted_opponent=True)
    model = PPO.load(model_path)
    for ep in range(episodes):
        o, _ = env.reset()
        done = False
        R = 0.0
        while not done:
            a, _ = model.predict(o['blue'], deterministic=True)
            o, r, term, trunc, _ = env.step({'blue': a})
            env.render()
            R += r['blue']
            done = term['blue'] or trunc['blue']
        print(f"Episode {ep+1}: return={R:.2f}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    run(args.model, args.episodes)
