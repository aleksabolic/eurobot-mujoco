import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
import mujoco
import mujoco.viewer

# -------------------- Constants --------------------
CTRL_DT = 0.02                       # Control period (50 Hz)
WORLD_TIMESTEP = None                # If None, uses model.opt.timestep
MAX_LIN_SPEED = 0.6                  # m/s (|v|<=1 scales to this)
MAX_ANG_SPEED = 2.0                  # rad/s (|w|<=1 scales to this)
PICK_RADIUS = 0.18                   # m
BOT_RADIUS = 0.12
TABLE_HALF = np.array([1.5, 1.0], dtype=np.float32)  
MARGIN = BOT_RADIUS 
WHEEL_RADIUS = 0.03 
AXLE_HALF    = 0.17 
CTRL_SUBSTEPS = None

# Top->Bottom, Left->Right
PANTRIES = {
    "A": np.array([-0.25, 0.45]),
    "B": np.array([ 0.25, 0.45]),
    "C": np.array([-1.4, -0.2]),
    "D": np.array([-0.7, -0.2]),
    "E": np.array([ 0.0, -0.2]),
    "F": np.array([ 0.7, -0.2]),
    "G": np.array([ 1.4, -0.2]),
    "H": np.array([-0.8, -0.9]),
    "I": np.array([ 0.0, -0.9]),
    "J": np.array([ 0.8, -0.9]),
}
PANTRY_R = 0.10

# Top->Bottom, Left->Right
PICKUPS = {
    "P1": np.array([-1.375, 0.2]),
    "P2": np.array([ 1.375, 0.2]),
    "P3": np.array([-0.35, -0.2]),
    "P4": np.array([ 0.35, -0.2]),
    "P5": np.array([-1.375, -0.6]),
    "P6": np.array([ 1.375, -0.6]),
    "P7": np.array([-0.4, -0.875]),
    "P8": np.array([ 0.4, -0.875]),
}
PICKUP_R = 0.075 

NESTS = {"blue": np.array([1.2, 0.8]),
         "yellow": np.array([-1.2,  0.8])}
NEST_HALF_SIZE = np.array([0.3, 0.225])  # half-widths (x,y)

AGENTS = ["blue", "yellow"]
ACT_SUFFIXES = ["x_act", "y_act", "yaw_act"]

COLOR_TO_INT = {"blue": 1, "yellow": 2}
INT_TO_COLOR = {1: "blue", 2: "yellow"}

def within_circle(p, center, r):
    return np.linalg.norm(p - center) <= r

def within_rect(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> bool:
    """Axis-aligned rectangle hit-test using center and half-sizes."""
    d = np.abs(p - center)
    return (d[0] <= half[0]) and (d[1] <= half[1])

class EurobotMJ(ParallelEnv):
    """
    Parallel Eurobot MJ environment.

    Observations per agent (float32 vector):
        [my_x, my_y, my_yaw, carrying_flag,
         opp_x, opp_y, opp_yaw,
         pickup occupancy,
         pantry occupancy,
         time_left_normalized]

    Actions per agent (Box):
        [v_scaled, w_scaled, op] where
          v_scaled in [-1,1] -> linear v in [-MAX_LIN_SPEED, +MAX_LIN_SPEED]
          w_scaled in [-1,1] -> angular w in [-MAX_ANG_SPEED, +MAX_ANG_SPEED]
          op in {0: noop, 1: pick, 2: drop}  (passed in as float, rounded)
    """
    metadata = {"name": "EurobotMJ-v0", "render_fps": int(1 / CTRL_DT)}

    def __init__(self,
                 xml_path: str = "assets/arena.xml",
                 max_steps: int = 2000,
                 scripted_opponent: bool = True):
        self.xml_path = xml_path
        self.max_steps = int(max_steps)
        self.scripted_opponent = bool(scripted_opponent)

        self.agents = AGENTS[:]  # ['blue', 'yellow']

        self.pickup_names = list(PICKUPS.keys())
        self.pantry_names = list(PANTRIES.keys())

        # Zone states (discrete)
        # pickup_occ[i] ∈ {0,1}: 1 means a full batch is available to pick.
        self.pickup_occ = np.ones(len(self.pickup_names), dtype=np.int32)

        # pantry_occ[i] ∈ {0,1,2}: 0 empty, 1 blue, 2 yellow (who currently holds it)
        self.pantry_occ = np.zeros(len(self.pantry_names), dtype=np.int32)

        # Carry state: -1 means not carrying, otherwise 1 (carrying a full batch)
        self.carry = {"blue": -1, "yellow": -1}

        # Load MJ model/data
        self._model = mujoco.MjModel.from_xml_path(self.xml_path)
        if WORLD_TIMESTEP is not None:
            self._model.opt.timestep = float(WORLD_TIMESTEP)
        self._data = mujoco.MjData(self._model)

        # Wheel motor actuator ids
        self._wheel_ids = {
            a: {
                "left":  self._model.actuator(f"{a}_left_motor").id,
                "right": self._model.actuator(f"{a}_right_motor").id,
            } for a in self.agents
        }

        # Suggested integration substeps per control step
        self._substeps = CTRL_SUBSTEPS or max(1, int(CTRL_DT / self._model.opt.timestep + 1e-9))

        self._site_ids = {a: self._model.site(f"{a}_grip").id for a in self.agents}

        self._pantry_geom_ids = {n: self._model.geom(f"pantry_{n}").id for n in self.pantry_names}
        self._pickup_geom_ids = {n: self._model.geom(f"pickup_{n}").id for n in self.pickup_names}

        # simple color table (R,G,B,A)
        self._CLR_EMPTY   = np.array([0.00, 0.80, 0.80, 0.35], dtype=np.float32)  # cyan = free
        self._CLR_BLUE    = np.array([0.20, 0.40, 1.00, 0.65], dtype=np.float32)  # owned by blue
        self._CLR_YELLOW  = np.array([1.00, 0.90, 0.20, 0.65], dtype=np.float32)  # owned by yellow
        self._CLR_AVAIL   = np.array([0.90, 0.10, 0.90, 0.35], dtype=np.float32)  # pickup has crate (magenta)
        self._CLR_SPENT   = np.array([0.30, 0.30, 0.30, 0.20], dtype=np.float32)  # pickup used/empty

        # Build spaces
        self._init_spaces()

        # State
        self._t = 0.0
        self._step_count = 0
        self.carry = {"blue": -1, "yellow": -1}  # crate index carried or -1
        self.rng = np.random.default_rng()
        self._viewer = None

    # -------------------- Spaces --------------------
    def _init_spaces(self):
        obs_dim = (4  # my (x, y, yaw, carrying_flag)
                   + 3  # opp (x, y, yaw)
                   + len(self.pickup_names) # pickup occupancy (0/1)
                   + len(self.pantry_names) # pantry occupancy (0/1/2)
                   + 1)  # time_left
        self.observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
            for a in self.agents
        }
        # [v, w, op] with op ∈ {0,1,2} but kept continuous for a simple API
        self.action_spaces = {
            a: spaces.Box(low=np.array([-1., -1., 0.], dtype=np.float32),
                          high=np.array([ 1.,  1., 2.], dtype=np.float32),
                          dtype=np.float32)
            for a in self.agents
        }

    # -------------------- MJ helpers --------------------
    def _qpos_index(self, joint_name: str) -> int:
        j = self._model.joint(joint_name).id
        return self._model.jnt_qposadr[j]

    def _body_xy(self, body_name: str) -> np.ndarray:
        bid = self._model.body(body_name).id
        return self._data.xpos[bid][:2].copy()

    def _body_yaw(self, body_name: str) -> float:
        bid = self._model.body(body_name).id
        m = self._data.xmat[bid].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def _set_body_pose2d(self, body_name: str, pos_xy: np.ndarray, yaw: float):
        """Set free2D (x, y, yaw) pose for a named body that has 3 planar joints."""
        bid = self._model.body(body_name).id
        adr = self._model.body_jntadr[bid]
        self._data.qpos[self._model.jnt_qposadr[adr + 0]] = float(pos_xy[0])
        self._data.qpos[self._model.jnt_qposadr[adr + 1]] = float(pos_xy[1])
        self._data.qpos[self._model.jnt_qposadr[adr + 2]] = float(yaw)

    def _set_agent_pose(self, agent: str, pos_xy: np.ndarray, yaw: float):
        # freejoint qpos: [x y z qw qx qy qz]
        jid = self._model.joint(f"{agent}_free").id
        adr = self._model.jnt_qposadr[jid]
        z = 0.035  # chassis height in XML; keep wheels in contact
        # yaw -> quaternion (z-rotation)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        qw, qx, qy, qz = cy, 0.0, 0.0, sy
        self._data.qpos[adr+0] = pos_xy[0]
        self._data.qpos[adr+1] = pos_xy[1]
        self._data.qpos[adr+2] = z
        self._data.qpos[adr+3] = qw
        self._data.qpos[adr+4] = qx
        self._data.qpos[adr+5] = qy
        self._data.qpos[adr+6] = qz


    def _refresh_markers(self):
        # Pantries: 0=empty, 1=blue, 2=yellow
        for i, name in enumerate(self.pantry_names):
            gid = self._pantry_geom_ids[name]
            occ = int(self.pantry_occ[i])
            if occ == 0:   color = self._CLR_EMPTY
            elif occ == 1: color = self._CLR_BLUE
            else:          color = self._CLR_YELLOW
            self._model.geom_rgba[gid] = color

        # Pickups: 1=available, 0=spent
        for i, name in enumerate(self.pickup_names):
            gid = self._pickup_geom_ids[name]
            color = self._CLR_AVAIL if int(self.pickup_occ[i]) == 1 else self._CLR_SPENT
            self._model.geom_rgba[gid] = color

    # -------------------- API: reset/step/render --------------------
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)
        self._t = 0.0
        self._step_count = 0
        self.pickup_occ[:] = 1
        self.pantry_occ[:] = 0
        self.carry = {"blue": -1, "yellow": -1}

        # Spawn robots
        self._set_agent_pose("blue",  NESTS["blue"], 0.0)
        self._set_agent_pose("yellow", NESTS["yellow"], 0.0)

        mujoco.mj_forward(self._model, self._data)
        self._refresh_markers()

        obs = {a: self._observe(a) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions: dict[str, np.ndarray]):
        # Blue from actions, Yellow scripted unless user provides both
        a_blue = actions["blue"]
        a_yellow = self._scripted_policy("yellow") if self.scripted_opponent else actions["yellow"]

        # Apply kinematic targets via position actuators
        self._apply_action("blue", a_blue)
        self._apply_action("yellow", a_yellow)

        self._step_count += 1
        self._t += CTRL_DT

        obs = {a: self._observe(a) for a in self.agents}
        rew_blue, rew_yellow = self._score_both()
        term = self._step_count >= self.max_steps
        rews = {"blue": rew_blue, "yellow": rew_yellow}
        terms = {"blue": term, "yellow": term}
        truncs = {"blue": False, "yellow": False}
        infos = {"blue": {}, "yellow": {}}
        return obs, rews, terms, truncs, infos

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        # just draw the current state; DO NOT step physics here
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    # -------------------- Observation --------------------
    def _ego(self, agent: str) -> tuple[np.ndarray, float, float]:
        """Return (my_xy, my_yaw, carrying_flag)."""
        p = self._body_xy(agent)
        th = self._body_yaw(agent)
        carrying = 1.0 if self.carry[agent] >= 0 else 0.0
        return p, th, carrying

    def _observe(self, agent: str) -> np.ndarray:
        me_idx = 0 if agent == "blue" else 1
        me, opp = self.agents[me_idx], self.agents[1 - me_idx]

        my_xy = self._body_xy(me)
        my_yaw = self._body_yaw(me)
        carrying = 1.0 if self.carry[me] >= 0 else 0.0

        opp_xy = self._body_xy(opp)
        opp_yaw = self._body_yaw(opp)

        # Normalize discrete occupancies to floats
        pickup_vec = self.pickup_occ.astype(np.float32)              # 0/1
        pantry_vec = self.pantry_occ.astype(np.float32)              # 0/1/2

        time_left = 1.0 - (self._step_count / self.max_steps)

        obs = np.concatenate([
            my_xy, [my_yaw, carrying],
            opp_xy, [opp_yaw],
            pickup_vec,
            pantry_vec,
            [time_left]
        ]).astype(np.float32)
        return obs


    # -------------------- Actions --------------------
    def _apply_action(self, agent: str, a: np.ndarray):
        # Parse command
        v = float(np.clip(a[0], -1, 1)) * MAX_LIN_SPEED
        w = float(np.clip(a[1], -1, 1)) * MAX_ANG_SPEED
        op = int(round(np.clip(a[2], 0, 2)))

        # Map (v, w) -> wheel angular velocities (rad/s)
        #   v_left  = (v - w*AXLE_HALF) / WHEEL_RADIUS
        #   v_right = (v + w*AXLE_HALF) / WHEEL_RADIUS
        v_left  = (v - w * AXLE_HALF) / WHEEL_RADIUS
        v_right = (v + w * AXLE_HALF) / WHEEL_RADIUS

        # Send setpoints to velocity actuators (ctrl = desired wheel speed)
        aidL = self._wheel_ids[agent]["left"]
        aidR = self._wheel_ids[agent]["right"]
        self._data.ctrl[aidL] = v_left
        self._data.ctrl[aidR] = v_right

        # Advance physics for one control period using substeps
        for _ in range(self._substeps):
            mujoco.mj_step(self._model, self._data)

        # After integration, we can safely check pick/drop zones (fresh positions)
        if op == 1:  # pick
            if self.carry[agent] <= 0:
                for i, name in enumerate(self.pickup_names):
                    if self.pickup_occ[i] == 1 and within_circle(self._grip_xy(agent), PICKUPS[name], PICKUP_R):
                        self.pickup_occ[i] = 0
                        self.carry[agent] = 1
                        self._refresh_markers()
                        break

        elif op == 2:  # drop
            if self.carry[agent] > 0:
                for i, name in enumerate(self.pantry_names):
                    if within_circle(self._body_xy(agent), PANTRIES[name], PANTRY_R):
                        self.pantry_occ[i] = COLOR_TO_INT[agent]
                        self.carry[agent] = -1
                        self._refresh_markers()
                        break



    def _grip_xy(self, agent: str) -> np.ndarray:
        return self._data.site_xpos[self._site_ids[agent]][:2].copy()

    # -------------------- Scoring --------------------
    def _score_agent(self, owner: str) -> float:
        owner_int = COLOR_TO_INT[owner]
        # Reward = number of pantries currently owned by this color
        return float(np.sum(self.pantry_occ == owner_int))


    def _score_both(self) -> tuple[float, float]:
        return self._score_agent("blue"), self._score_agent("yellow")

    # -------------------- Simple scripted policy --------------------
    def _scripted_policy(self, agent: str) -> np.ndarray:
        p = self._body_xy(agent)
        th = self._body_yaw(agent)

        if self.carry[agent] <= 0:
            # Go to nearest available pickup
            candidates = [
                (i, name, PICKUPS[name]) for i, name in enumerate(self.pickup_names)
                if self.pickup_occ[i] == 1
            ]
            if candidates:
                i, _, tgt = min(candidates, key=lambda t: np.linalg.norm(t[2] - p))
                op = 1 if within_circle(self._grip_xy(agent), tgt, PICKUP_R) else 0
            else:
                tgt = p  # nothing to do
                op = 0
        else:
            # Carrying: go to pantry not owned by me (prefer empty)
            my_int = COLOR_TO_INT[agent]
            # Prefer empty, else flip opponent-owned, else stay
            empty = [(i, name, PANTRIES[name]) for i, name in enumerate(self.pantry_names) if self.pantry_occ[i] == 0]
            opp_owned = [(i, name, PANTRIES[name]) for i, name in enumerate(self.pantry_names) if self.pantry_occ[i] not in (0, my_int)]

            pool = empty if empty else opp_owned
            if pool:
                i, _, tgt = min(pool, key=lambda t: np.linalg.norm(t[2] - p))
                op = 2 if within_circle(p, tgt, PANTRY_R) else 0
            else:
                tgt = p
                op = 0

        desired = np.arctan2((tgt - p)[1], (tgt - p)[0])
        w_cmd = np.clip((desired - th + np.pi) % (2 * np.pi) - np.pi, -MAX_ANG_SPEED, MAX_ANG_SPEED)

        ang_err = ((np.arctan2((tgt - p)[1], (tgt - p)[0]) - th + np.pi) % (2*np.pi)) - np.pi
        # mild P gain + deadzone
        if abs(ang_err) < 0.05:
            w_cmd = 0.0
        else:
            w_cmd = np.clip(0.6 * ang_err, -MAX_ANG_SPEED, MAX_ANG_SPEED)
        return np.array([1.0, w_cmd / MAX_ANG_SPEED, float(op)], dtype=np.float32)


