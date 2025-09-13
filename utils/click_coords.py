# click_coords.py
import argparse, numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to table image (png/jpg).")
    p.add_argument("--width",  type=float, required=True, help="Table width in meters (X).")
    p.add_argument("--height", type=float, required=True, help="Table height in meters (Y).")
    p.add_argument("--calibrate", action="store_true",
                   help="Click bottom-left then top-right corners to calibrate.")
    return p.parse_args()

def main():
    args = parse_args()
    img = mpimg.imread(args.image)
    H, W = img.shape[0], img.shape[1]

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click points. (Optional) If --calibrate: first click BL, then TR.\nClose window to quit.")
    ax.set_axis_off()

    calib = {"bl": (0, H-1), "tr": (W-1, 0)}  # defaults if not calibrating
    clicks = []

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if args.calibrate and len(clicks) < 2:
            clicks.append((x, y))
            if len(clicks) == 1:
                print(f"[calib] BL set at pixel ({x:.1f}, {y:.1f})")
            elif len(clicks) == 2:
                print(f"[calib] TR set at pixel ({x:.1f}, {y:.1f})")
                blx, bly = clicks[0]
                trx, try_ = clicks[1]
                # basic sanity: swap if misordered
                if trx < blx: blx, trx = trx, blx
                if try_ > bly: bly, try_ = try_, bly
                calib["bl"] = (blx, bly)
                calib["tr"] = (trx, try_)
            return

        blx, bly = calib["bl"]
        trx, try_ = calib["tr"]

        # Normalize to [0,1] in table frame with origin at BL, y up
        denom_x = (trx - blx)
        denom_y = (bly - try_)
        if denom_x == 0 or denom_y == 0:
            print("[error] Degenerate calibration box.")
            return

        u = (x - blx) / denom_x                    # 0 at BL.x → 1 at TR.x
        v = (bly - y) / denom_y                    # 0 at BL.y (bottom) → 1 at TR.y (top)
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        # Eurobot coords: origin bottom-left (meters)
        x_euro = u * args.width
        y_euro = v * args.height

        # MuJoCo coords: origin table center (meters)
        x_mj = x_euro - args.width  / 2.0
        y_mj = y_euro - args.height / 2.0

        print(f"px=({x:.1f},{y:.1f})  |  Eurobot=({x_euro:.3f},{y_euro:.3f}) m  |  MuJoCo=({x_mj:.3f},{y_mj:.3f}) m")

        # tiny dot for visual feedback
        ax.plot([x], [y], marker="o", ms=4, mfc="none", mec="r")
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

if __name__ == "__main__":
    main()
