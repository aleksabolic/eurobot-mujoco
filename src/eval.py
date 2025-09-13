import argparse, numpy as np
from stable_baselines3 import PPO
from eurobot_env import EurobotMJ, CTRL_DT
import time 

def run(model_path, episodes=5, render=False):
    env = EurobotMJ(xml_path="assets/arena.xml", max_steps=2000, scripted_opponent=True)
    model = PPO.load(model_path)
    for ep in range(episodes):
        o, _ = env.reset()
        done = False
        R = 0.0
        R_y = 0.0
        while not done:
            a, _ = model.predict(o['blue'], deterministic=True)
            o, r, term, trunc, _ = env.step({'blue': a})
            if render:
                env.render()
                time.sleep(CTRL_DT)
            R += r['blue']
            R_y += r['yellow']
            done = term['blue'] or trunc['blue']
        print(f"Episode {ep+1}: return blue={R:.2f} | return yellow={R_y:.2f}")
    if render:
        env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    run(args.model, args.episodes, args.render)
