import argparse
import os
from stable_baselines3 import PPO
from eurobot_env import EurobotDiscreteEnv
from eurobot_render import EurobotCV2Renderer, review
from robot import RobotProfile
import time
from masked_policy import MaskedMultiCatPolicy
import cv2

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

def run(model_path, episodes=1, fps=1.0, gif_path=None):
    env = EurobotDiscreteEnv()
    model = PPO.load(model_path, device="auto", custom_objects={"policy_class": MaskedMultiCatPolicy})
    renderer = EurobotCV2Renderer(env.world, size=(1000, 700)) 
    
    for ep in range(episodes):
        o, _ = env.reset()
        done = False
        rl_return = 0.0
        blue_score = None
        yellow_score = None
        frames = [] if gif_path else None
        while not done:
            a, _ = model.predict(o, deterministic=True)
            o, r, term, _, _ = env.step(a)
            rl_return += r
            done = term
            if gif_path:
                img_bgr = renderer.draw_snapshot(show=False)
                frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
        blue_score, yellow_score = env.world.final_scores()
        print(
            f"Episode {ep+1}: rl_return_blue={rl_return:.2f} "
            f"score_blue={blue_score:.2f} score_yellow={yellow_score:.2f}"
        )
        print(env.world.last_invalid_detail)
        print([h["tag"] for h in env.world.history[:12]])
        print([h["tag"] for h in env.world.history[-12:]])
        if frames is not None:
            if imageio is None:
                raise RuntimeError("imageio is required for GIF export. pip install imageio")
            if episodes == 1:
                out_path = gif_path if os.path.splitext(gif_path)[1] else gif_path + ".gif"
            else:
                root, ext = os.path.splitext(gif_path)
                if not ext:
                    ext = ".gif"
                out_path = f"{root}_ep{ep+1}{ext}"
            imageio.mimsave(out_path, frames, duration=1.0/max(fps, 1.0))
            print(f"Saved GIF: {out_path}")
            review(frames)

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--gif", type=str, default="runs/eval.gif", help="Path to save GIF")
    args = ap.parse_args()
    run(args.model, args.episodes, args.fps, args.gif)
