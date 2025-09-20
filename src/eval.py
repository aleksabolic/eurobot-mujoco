import argparse
from stable_baselines3 import PPO
from eurobot_env import EurobotDiscreteEnv
from eurobot_render import EurobotCV2Renderer
from robot import RobotProfile
import time

def run(model_path, episodes=3, render=False):
    env = EurobotDiscreteEnv()
    model = PPO.load(model_path, device="auto")
    renderer = EurobotCV2Renderer(env.world, size=(1000, 700)) 
    for ep in range(episodes):
        o, _ = env.reset()
        done = False
        R = 0.0
        while not done:
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, _ = env.step(a)
            R += r
            done = term
            if render:
                renderer.draw_snapshot(show=True)
                time.sleep(1/10.0)
        print(f"Episode {ep+1}: return blue={R:.2f}")
        print([h["tag"] for h in env.world.history[:12]])
        print([h["tag"] for h in env.world.history[-12:]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    run(args.model, args.episodes, args.render)
