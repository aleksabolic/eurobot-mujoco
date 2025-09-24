# eurobot_render_cv2.py
import numpy as np, cv2
from pathlib import Path
from eurobot_world import EurobotWorld, NodeType, Col

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
FACE_NEUTRAL = (242,242,217)
FACE_AVAIL   = (242,191,242)
FACE_EMPTY   = (217,217,217)

class EurobotCV2Renderer:
    def __init__(self, world: EurobotWorld, size=(900, 600), background_path: Path | None = None, background_alpha: float = 0.35):
        self.world = world
        self.W, self.H = size
        self.pad_left = 20
        self.pad_right = 20
        self.pad_bottom = 20
        self.pad_top = 80   # reserve space for info banner
        avail_w = self.W - self.pad_left - self.pad_right
        avail_h = self.H - self.pad_top - self.pad_bottom
        self.scale = min(avail_w / (TABLE_X_MAX - TABLE_X_MIN),
                         avail_h / (TABLE_Y_MAX - TABLE_Y_MIN))
        self.ox = self.pad_left - TABLE_X_MIN * self.scale
        self.oy = self.pad_bottom - TABLE_Y_MIN * self.scale
        self.board_w = int(round((TABLE_X_MAX - TABLE_X_MIN) * self.scale))
        self.board_h = int(round((TABLE_Y_MAX - TABLE_Y_MIN) * self.scale))
        self.board_left = int(self.pad_left)
        self.board_top = int(self.pad_top)

        self.table_img = None
        self.bg_alpha = float(background_alpha)
        if background_path is None:
            background_path = Path(__file__).resolve().parents[1] / "assets" / "table_bis.png"
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

        # nests
        for node in self.world.nodes:
            if node.kind == NodeType.NEST:
                cx,cy = node.xy
                p0 = self._w2p(cx - NEST_HALF_X, cy - NEST_HALF_Y)
                p1 = self._w2p(cx + NEST_HALF_X, cy + NEST_HALF_Y)
                cv2.rectangle(img, p0, p1, BLK, 1)

        # snapshot
        if self.world.history:
            s = self.world.history[idx]
            blue_node, yellow_node = s["blue_node"], s["yellow_node"]
            pantries, pickups = s["pantries"], s["pickups"]
            blue_inv, yellow_inv = s["blue_inv"], s["yellow_inv"]
            t_left = s["t_left"]
        else:
            blue_node, yellow_node = self.world.blue.node, self.world.yellow.node
            pantries, pickups = self.world.pantries, self.world.pickups
            blue_inv, yellow_inv = self.world.blue.inv, self.world.yellow.inv
            t_left = self.world.t_left

        # pantries
        for k, idx_node in enumerate(self.world.PANTRIES):
            node = self.world.nodes[idx_node]
            cx, cy = node.xy
            p0 = self._w2p(cx - PANTRY_HALF, cy - PANTRY_HALF)
            p1 = self._w2p(cx + PANTRY_HALF, cy + PANTRY_HALF)
            b, y, n = int(pantries[k, Col.BLUE]), int(pantries[k, Col.YELLOW]), int(pantries[k, Col.NEUTRAL])
            if b>y: c=FACE_BLUE
            elif y>b: c=FACE_YELLOW
            else: c=FACE_NEUTRAL
            cv2.rectangle(img, p0, p1, c, thickness=-1)
            cv2.rectangle(img, p0, p1, BLK, 1)
            label = f"B{b}/Y{y}/N{n}"
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
        cv2.circle(img, b, R, (255,102,51), -1)
        cv2.circle(img, b, R, BLK, 2)
        cv2.circle(img, y, R, (51,230,255), -1)
        cv2.circle(img, y, R, BLK, 2)

        # header
        header_x = self.pad_left
        header_y = 24
        cv2.putText(img, f"t_left = {t_left:.1f}s", (header_x, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLK, 2, cv2.LINE_AA)
        cv2.putText(img, f"BLUE inv: B{int(blue_inv[Col.BLUE])} Y{int(blue_inv[Col.YELLOW])} N{int(blue_inv[Col.NEUTRAL])}",
                    (header_x, header_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv2.LINE_AA)
        cv2.putText(img, f"YELL inv: B{int(yellow_inv[Col.BLUE])} Y{int(yellow_inv[Col.YELLOW])} N{int(yellow_inv[Col.NEUTRAL])}",
                    (header_x, header_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv2.LINE_AA)

        if show:
            cv2.imshow("Eurobot (cv2)", img); cv2.waitKey(1)
        return img  # you can pipe this to a VideoWriter
