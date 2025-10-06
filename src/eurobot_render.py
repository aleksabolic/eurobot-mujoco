# eurobot_render_cv2.py
import numpy as np, cv2
from pathlib import Path
from eurobot_world import EurobotWorld, NodeType, Col

try:
    import tkinter as _tk
except Exception:
    _tk = None

TABLE_X_MIN, TABLE_X_MAX = -1.5, 1.5
TABLE_Y_MIN, TABLE_Y_MAX = -1.0, 1.0
PANTRY_HALF = 0.10
PICKUP_R    = 0.075
NEST_HALF_X = 0.30
NEST_HALF_Y = 0.225

BLUE = (255,102,51)      # BGR
YELL = (51,230,255)
BLK  = (0,0,0)
WHT  = (255,255,255)
FACE_BLUE    = (255,210,191)
FACE_YELLOW  = (179,250,255)
FACE_TIED    = (242,242,217)
FACE_AVAIL   = (242,191,242)
FACE_EMPTY   = (217,217,217)

class EurobotCV2Renderer:
    def __init__(self, world: EurobotWorld, size=(900, 600), background_path: Path | None = None, background_alpha: float = 0.35):
        self.world = world
        self.W, self.H = size
        self._win_title = "Eurobot (cv2)"
        self._win_created = False
        self._win_centered = False
        self.pad_left = 20
        self.pad_right = 20
        self.pad_bottom = 20
        self.pad_top = 80   # reserve space for info banner
        avail_w = self.W - self.pad_left - self.pad_right
        avail_h = self.H - self.pad_top - self.pad_bottom
        self.scale = min(avail_w / (TABLE_X_MAX - TABLE_X_MIN),
                         avail_h / (TABLE_Y_MAX - TABLE_Y_MIN))
        
        self.board_w = int(round((TABLE_X_MAX - TABLE_X_MIN) * self.scale))
        self.board_h = int(round((TABLE_Y_MAX - TABLE_Y_MIN) * self.scale)) 
        dx = max(0, (avail_w - self.board_w) // 2)
        dy = max(0, (avail_h - self.board_h) // 2)
        self.board_left = int(self.pad_left + dx)
        self.board_top  = int(self.pad_top  + dy)

        self.ox = self.board_left - TABLE_X_MIN * self.scale
        self.oy = (self.H - (self.board_top + self.board_h)) - TABLE_Y_MIN * self.scale

        self.table_img = None
        self.bg_alpha = float(background_alpha)
        if background_path is None:
            background_path = Path(__file__).resolve().parents[1] / "assets" / "table.png"
        try:
            if background_path.exists():
                img = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
                if img is not None and self.board_w > 0 and self.board_h > 0:
                    self.table_img = cv2.resize(img, (self.board_w, self.board_h), interpolation=cv2.INTER_AREA)
        except Exception:
            self.table_img = None

    def _w2p(self, x, y):
        px = int(self.ox + x * self.scale)
        py = int(self.H - (self.oy + y * self.scale))   # invert Y
        return px, py


    def _l2p(self, L):
        return int(round(L*self.scale))

    def _scores_from_snapshot(self, pantries, nest_blue: int, nest_yellow: int,
                               blue_node: int, yellow_node: int) -> tuple[float, float]:
        pan = np.asarray(pantries)
        if pan.ndim == 0:
            pan = pan.reshape(0, len(Col))
        if pan.ndim == 1:
            pan = pan.reshape(-1, len(Col))

        rewards = self.world.rewards
        blue_pantry = int(pan[:, Col.BLUE].sum()) if pan.size else 0
        yellow_pantry = int(pan[:, Col.YELLOW].sum()) if pan.size else 0

        blue_score = rewards.pantry_bonus * blue_pantry + rewards.nest_bonus * int(nest_blue)
        yellow_score = rewards.pantry_bonus * yellow_pantry + rewards.nest_bonus * int(nest_yellow)

        for row in pan:
            b = int(row[Col.BLUE])
            y = int(row[Col.YELLOW])
            if b > y:
                blue_score += rewards.interest_bonus
            elif y > b:
                yellow_score += rewards.interest_bonus

        if int(blue_node) == self.world.NEST_BLUE:
            blue_score += rewards.finish_in_nest_bonus
        if int(yellow_node) == self.world.NEST_YELL:
            yellow_score += rewards.finish_in_nest_bonus

        return float(blue_score), float(yellow_score)

    def draw_snapshot(self, idx=-1, show=False):
        img = np.full((self.H, self.W, 3), 255, np.uint8)
        # info banner background
        cv2.rectangle(img, (0, 0), (self.W, self.pad_top - 10), (240, 240, 240), -1)

        if self.table_img is not None:
            h = min(self.table_img.shape[0], self.H - self.board_top)
            w = min(self.table_img.shape[1], self.W - self.board_left)
            if h > 0 and w > 0:
                roi = img[self.board_top:self.board_top + h, self.board_left:self.board_left + w]
                overlay = self.table_img[:h, :w]
                cv2.addWeighted(overlay, self.bg_alpha, roi, 1 - self.bg_alpha, 0, roi)

        # border
        x0,y0 = self._w2p(TABLE_X_MIN, TABLE_Y_MIN)
        x1,y1 = self._w2p(TABLE_X_MAX, TABLE_Y_MAX)
        cv2.rectangle(img, (x0,y0), (x1,y1), BLK, 1)

        # snapshot
        if self.world.history:
            s = self.world.history[idx]
            blue_node, yellow_node = s["blue_node"], s["yellow_node"]
            pantries, pickups = s["pantries"], s["pickups"]
            blue_inv, yellow_inv = s["blue_inv"], s["yellow_inv"]
            t_left = s["t_left"]
            nest_blue = int(s.get("nest_blue", self.world.nest_blue_counted))
            nest_yellow = int(s.get("nest_yellow", self.world.nest_yellow_counted))
            if "blue_score" in s and "yellow_score" in s:
                blue_score = float(s["blue_score"])
                yellow_score = float(s["yellow_score"])
            else:
                blue_score, yellow_score = self._scores_from_snapshot(
                    pantries, nest_blue, nest_yellow, blue_node, yellow_node
                )
            blue_return = float(s.get("blue_return", getattr(self.world, "blue_return", 0.0)))
        else:
            blue_node, yellow_node = self.world.blue.node, self.world.yellow.node
            pantries, pickups = self.world.pantries, self.world.pickups
            blue_inv, yellow_inv = self.world.blue.inv, self.world.yellow.inv
            t_left = self.world.t_left
            nest_blue = int(self.world.nest_blue_counted)
            nest_yellow = int(self.world.nest_yellow_counted)
            blue_score, yellow_score = self.world.final_scores()
            blue_return = float(getattr(self.world, "blue_return", 0.0))

        # nests
        label_pad = 4
        for node in self.world.nodes:
            if node.kind == NodeType.NEST:
                cx, cy = node.xy
                p0 = self._w2p(cx - NEST_HALF_X, cy - NEST_HALF_Y)
                p1 = self._w2p(cx + NEST_HALF_X, cy + NEST_HALF_Y)
                tl = (min(p0[0], p1[0]), min(p0[1], p1[1]))
                br = (max(p0[0], p1[0]), max(p0[1], p1[1]))
                cv2.rectangle(img, tl, br, BLK, 1)

                if node.name == "NestBlue":
                    fill = FACE_BLUE
                    label = f"B{nest_blue}"
                else:
                    fill = FACE_YELLOW
                    label = f"Y{nest_yellow}"

                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                box_tl = (tl[0] + label_pad, tl[1] + label_pad)
                box_br = (box_tl[0] + tw + label_pad * 2, box_tl[1] + th + baseline + label_pad)
                cv2.rectangle(img, box_tl, box_br, fill, thickness=-1)

                text_x = box_tl[0] + label_pad
                text_y = box_br[1] - label_pad - baseline
                cv2.putText(img, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv2.LINE_AA)

        # pantries
        for k, idx_node in enumerate(self.world.PANTRIES):
            node = self.world.nodes[idx_node]
            cx, cy = node.xy
            p0 = self._w2p(cx - PANTRY_HALF, cy - PANTRY_HALF)
            p1 = self._w2p(cx + PANTRY_HALF, cy + PANTRY_HALF)
            b, y = int(pantries[k, Col.BLUE]), int(pantries[k, Col.YELLOW])
            if b>y: c=FACE_BLUE
            elif y>b: c=FACE_YELLOW
            else: c=FACE_TIED
            cv2.rectangle(img, p0, p1, c, thickness=-1)
            cv2.rectangle(img, p0, p1, BLK, 1)
            label = f"B{b}/Y{y}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = p0[0] + ((p1[0] - p0[0] - tw) // 2)
            text_y = p0[1] + ((p1[1] - p0[1] + th) // 2)
            cv2.putText(img, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLK, 1, cv2.LINE_AA)

        # pickups
        for k, idx_node in enumerate(self.world.PICKUPS):
            node = self.world.nodes[idx_node]
            cx, cy = node.xy
            px, py = self._w2p(cx, cy)
            r = self._l2p(PICKUP_R)
            b, y = int(pickups[k, Col.BLUE]), int(pickups[k, Col.YELLOW])
            c = FACE_AVAIL if (b+y)>0 else FACE_EMPTY
            cv2.circle(img, (px,py), r, c, -1)
            cv2.circle(img, (px,py), r, BLK, 1)
            label = f"B{b} Y{y}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = px - tw // 2
            text_y = py + th // 2
            cv2.putText(img, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLK, 1, cv2.LINE_AA)

        # robots
        bx, by = self.world.nodes[blue_node].xy
        yx, yy = self.world.nodes[yellow_node].xy
        b = self._w2p(bx, by); y = self._w2p(yx, yy)
        R = self._l2p(0.12)
        overlay = img.copy()
        robot_alpha = 0.55
        cv2.circle(overlay, b, R, (255,102,51), -1)
        cv2.circle(overlay, y, R, (51,230,255), -1)
        cv2.addWeighted(overlay, robot_alpha, img, 1 - robot_alpha, 0, img)
        cv2.circle(img, b, R, (255,102,51), 2)
        cv2.circle(img, b, R, BLK, 2)
        cv2.circle(img, y, R, (51,230,255), 2)
        cv2.circle(img, y, R, BLK, 2)

        # header
        header_x = self.pad_left
        header_y = 24
        cv2.putText(img, f"t_left = {t_left:.1f}s", (header_x, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLK, 2, cv2.LINE_AA)
        cv2.putText(img, f"BLUE inv: B{int(blue_inv[Col.BLUE])} Y{int(blue_inv[Col.YELLOW])}",
                    (header_x, header_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv2.LINE_AA)
        cv2.putText(img, f"YELL inv: B{int(yellow_inv[Col.BLUE])} Y{int(yellow_inv[Col.YELLOW])}",
                    (header_x, header_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv2.LINE_AA)

        score_x = self.pad_left + 320
        cv2.putText(img, f"BLUE score: {blue_score:.1f}", (score_x, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv2.LINE_AA)
        cv2.putText(img, f"YELL score: {yellow_score:.1f}", (score_x, header_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv2.LINE_AA)
        cv2.putText(img, f"BLUE RL return: {blue_return:.3f}", (score_x, header_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv2.LINE_AA)

        if show:
            # create the window once so we can move it
            if not self._win_created:
                # WINDOW_AUTOSIZE uses image size; GUI_NORMAL allows resizing (both are fine)
                cv2.namedWindow(self._win_title, cv2.WINDOW_AUTOSIZE)
                self._win_created = True

            cv2.imshow(self._win_title, img)
            cv2.waitKey(1)

            # center only on the first visible frame
            if not self._win_centered:
                try:
                    x, y, w, h = cv2.getWindowImageRect(self._win_title)  # OpenCV >=4.5
                    sw, sh = self._screen_size()
                    nx = max(0, (sw - w) // 2)
                    ny = max(0, (sh - h) // 2)
                    cv2.moveWindow(self._win_title, nx, ny)
                    self._win_centered = True
                except Exception:
                    # if getWindowImageRect isn't available, still attempt to move using image size
                    sw, sh = self._screen_size()
                    nx = max(0, (sw - self.W) // 2)
                    ny = max(0, (sh - self.H) // 2)
                    cv2.moveWindow(self._win_title, nx, ny)
                    self._win_centered = True

        return img


    def _screen_size(self) -> tuple[int, int]:
        """Return (screen_w, screen_h) using Tk if available, else fall back to image size."""
        if _tk is not None:
            try:
                root = _tk.Tk()
                root.withdraw()
                w = root.winfo_screenwidth()
                h = root.winfo_screenheight()
                root.destroy()
                if w > 0 and h > 0:
                    return int(w), int(h)
            except Exception:
                pass
        # fallback: something sane
        return 1920, 1080

# Helper for reviewing the gif
def review(frames_rgb, title="Eurobot (cv2)", fps=5):
    assert len(frames_rgb) > 0, "No frames to review"

    import tkinter as _tk
    def _screen_size():
        try:
            root = _tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            if w > 0 and h > 0:
                return int(w), int(h)
        except Exception:
            pass
        return 1920, 1080

    H, W = frames_rgb[0].shape[:2]
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # make sure window exists before moving

    try:
        x, y, w, h = cv2.getWindowImageRect(title)
        sw, sh = _screen_size()
        nx = max(0, (sw - w) // 2)
        ny = max(0, (sh - h) // 2)
        cv2.moveWindow(title, nx, ny)
    except Exception:
        sw, sh = _screen_size()
        nx = max(0, (sw - W) // 2)
        ny = max(0, (sh - H) // 2)
        cv2.moveWindow(title, nx, ny)

    n = len(frames_rgb)
    idx = 0
    playing = True
    updating_from_code = False
    delay = max(1, int(1000 / max(fps, 1.0)))

    def show(i):
        img = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        cv2.imshow(title, img)

    def on_trackbar(v):
        nonlocal idx, playing
        idx = int(v)
        show(idx)
        if not updating_from_code:
            playing = False

    cv2.createTrackbar("t", title, 0, n - 1, on_trackbar)

    def get_key(delay_ms):
        k = cv2.waitKeyEx(delay_ms) & 0xFFFFFFFF
        if k == 0xFFFFFFFF: return ""
        if k in (ord('q'), ord('Q')): return "quit"
        if k == 27: return "esc"              # don't quit on 'esc' unless you want to
        if k == 32: return "space"

        # Linux/X11
        if k == 65361: return "left"
        if k == 65363: return "right"
        if k == 65360: return "home"
        if k == 65367: return "end"
        # Windows waitKeyEx
        if k == 2424832: return "left"
        if k == 2555904: return "right"
        if k == 2359296: return "home"
        if k == 2293760: return "end"
        return ""

    while True:
        if playing:
            if idx < n - 1: idx += 1
            else: playing = False
            updating_from_code = True; cv2.setTrackbarPos("t", title, idx); updating_from_code = False
            show(idx)

        key = get_key(delay)
        if not key: 
            continue
        if key in ("quit",): 
            break
        if key == "space":
            playing = not playing
        elif key == "left":
            playing = False
            idx = max(0, idx - 1)
            updating_from_code = True; cv2.setTrackbarPos("t", title, idx); updating_from_code = False
            show(idx)
        elif key == "right":
            playing = False
            idx = min(n - 1, idx + 1)
            updating_from_code = True; cv2.setTrackbarPos("t", title, idx); updating_from_code = False
            show(idx)
        elif key == "home":
            playing = False
            idx = 0
            updating_from_code = True; cv2.setTrackbarPos("t", title, idx); updating_from_code = False
            show(idx)
        elif key == "end":
            playing = False
            idx = n - 1
            updating_from_code = True; cv2.setTrackbarPos("t", title, idx); updating_from_code = False
            show(idx)
        elif key == "esc":
            # optional: treat ESC as quit; if not, ignore it so arrows on some systems don't kill the app
            pass

    cv2.destroyWindow(title)