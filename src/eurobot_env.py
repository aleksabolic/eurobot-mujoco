import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
import mujoco
import mujoco.viewer

# -------------------- Constants --------------------
CTRL_DT = 0.02                       # Control period (50 Hz)
WORLD_TIMESTEP = None                # If None, uses model.opt.timestep
MAX_LIN_SPEED = 0.6                  # m/s (|v|<=1 scales to this)
MAX_ANG_SPEED = 1.2                  # rad/s (|w|<=1 scales to this)
WHEEL_RADIUS = 0.06   
AXLE_HALF    = 0.11   
BASE_Z       = 0.12   
CTRL_SUBSTEPS = None

# Top->Bottom, Left->Right
PANTRIES = {
    "A": np.array([-0.25,  0.45]),   # unchanged
    "B": np.array([ 0.25,  0.45]),   # unchanged
    "C": np.array([-1.32, -0.20]),   # was -1.40
    "D": np.array([-0.70, -0.20]),
    "E": np.array([ 0.00, -0.20]),
    "F": np.array([ 0.70, -0.20]),
    "G": np.array([ 1.32, -0.20]),   # was  1.40
    "H": np.array([-0.80, -0.82]),   # was -0.90
    "I": np.array([ 0.00, -0.82]),   # was -0.90
    "J": np.array([ 0.80, -0.82]),   # was -0.90
}
PANTRY_R = 0.10

# Top->Bottom, Left->Right
PICKUPS = {
    "P1": np.array([-1.32,  0.20]),  # was -1.375
    "P2": np.array([ 1.32,  0.20]),  # was  1.375
    "P3": np.array([-0.35, -0.20]),
    "P4": np.array([ 0.35, -0.20]),
    "P5": np.array([-1.32, -0.60]),  # was -1.375
    "P6": np.array([ 1.32, -0.60]),  # was  1.375
    "P7": np.array([-0.40, -0.82]),  # was -0.875
    "P8": np.array([ 0.40, -0.82]),  # was -0.875
}
PICKUP_R = 0.075 

NESTS = {"blue": np.array([1.2, 0.8]),
         "yellow": np.array([-1.2,  0.8])}
NEST_HALF_SIZE = np.array([0.3, 0.225])  # half-widths (x,y)

AGENTS = ["blue", "yellow"]

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


        self._BODY = {"blue": "blue_base", "yellow": "yellow_base"}
        self._FREE = {"blue": "blue_base_free", "yellow": "yellow_base_free"}
        self._ACT  = {
            "blue":   {"left": "blue_left_speed",   "right": "blue_right_speed"},
            "yellow": {"left": "yellow_left_speed", "right": "yellow_right_speed"},
        }
        self._SITE = {"blue": "blue_grip", "yellow": "yellow_grip"}

        # Wheel motor actuator ids
        self._wheel_ids = {
            a: {
                "left":  self._model.actuator(self._ACT[a]["left"]).id,
                "right": self._model.actuator(self._ACT[a]["right"]).id,
            } for a in self.agents
        }

        # --- IDs for wall/blue contact checks ---
        self._wall_geom_ids = {
            self._model.geom("wall_north").id,
            self._model.geom("wall_south").id,
            self._model.geom("wall_east").id,
            self._model.geom("wall_west").id,
        }
        # Parts of blue that can touch walls
        self._blue_contact_geom_ids = {
            self._model.geom("blue_chassis").id,
            self._model.geom("blue_left_tire").id,
            self._model.geom("blue_right_tire").id,
            self._model.geom("blue_caster_front").id,
            self._model.geom("blue_caster_back").id,
            # add more if you attach bumpers/claws later
        }

        # Suggested integration substeps per control step
        self._substeps = CTRL_SUBSTEPS or max(1, int(CTRL_DT / self._model.opt.timestep + 1e-9))

        self._site_ids = {a: self._model.site(self._SITE[a]).id for a in self.agents}

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

    def _body_xy(self, agent: str) -> np.ndarray:
        bid = self._model.body(self._BODY[agent]).id
        return self._data.xpos[bid][:2].copy()

    def _body_yaw(self, agent: str) -> float:
        bid = self._model.body(self._BODY[agent]).id
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
        jid = self._model.joint(self._FREE[agent]).id
        adr = self._model.jnt_qposadr[jid]
        z = BASE_Z
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

    def _blue_wall_contact(self) -> tuple[bool, int]:
        """Return (hit_any_wall, num_contacts) for blue vs walls this step."""
        n_hit = 0
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if ((g1 in self._blue_contact_geom_ids and g2 in self._wall_geom_ids) or
                (g2 in self._blue_contact_geom_ids and g1 in self._wall_geom_ids)):
                n_hit += 1
        return (n_hit > 0), n_hit
    
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
        self._set_agent_pose("blue",  NESTS["blue"], -np.pi/2)
        self._set_agent_pose("yellow", NESTS["yellow"], -np.pi/2)

        mujoco.mj_forward(self._model, self._data)
        self._refresh_markers()

        obs = {a: self._observe(a) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions: dict[str, np.ndarray]):
        # Decide both actions first
        a_blue = actions["blue"]
        a_yellow = self._scripted_policy("yellow") if self.scripted_opponent else actions["yellow"]

        # Apply controls (no stepping yet)
        self._apply_action("blue", a_blue)
        self._apply_action("yellow", a_yellow)

        # Advance physics ONCE per env step
        for _ in range(self._substeps):
            mujoco.mj_step(self._model, self._data)

        # After integration, resolve pick/drop (fresh poses)
        self._maybe_pick_or_drop("blue")
        self._maybe_pick_or_drop("yellow")

        # Update visuals
        self._refresh_markers()

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
    def _apply_action(self, agent: str, a: np.ndarray) -> None:
        # Parse command
        v = float(np.clip(a[0], -1, 1)) * MAX_LIN_SPEED
        w = float(np.clip(a[1], -1, 1)) * MAX_ANG_SPEED
        # op = int(round(np.clip(a[2], 0, 2)))  # IGNORED now

        # Map (v, w) -> wheel angular velocities (rad/s)
        v_left  = (v - w * AXLE_HALF) / WHEEL_RADIUS
        v_right = (v + w * AXLE_HALF) / WHEEL_RADIUS

        # Send setpoints
        aidL = self._wheel_ids[agent]["left"]
        aidR = self._wheel_ids[agent]["right"]
        self._data.ctrl[aidL] = v_left
        self._data.ctrl[aidR] = v_right

    def _maybe_pick_or_drop(self, agent: str):
        # Auto-pick
        if self.carry[agent] < 0:
            grip = self._grip_xy(agent)
            for i, name in enumerate(self.pickup_names):
                if self.pickup_occ[i] == 1 and within_circle(grip, PICKUPS[name], PICKUP_R):
                    self.pickup_occ[i] = 0
                    self.carry[agent] = 1
                    break
        # Auto-drop
        else:
            p = self._body_xy(agent)
            for i, name in enumerate(self.pantry_names):
                if within_circle(p, PANTRIES[name], PANTRY_R):
                    self.pantry_occ[i] = COLOR_TO_INT[agent]
                    self.carry[agent] = -1
                    break

    def _grip_xy(self, agent: str) -> np.ndarray:
        return self._data.site_xpos[self._site_ids[agent]][:2].copy()

    # -------------------- Scoring --------------------
    def _score_agent(self, owner: str) -> float:
        owner_int = COLOR_TO_INT[owner]
        # Reward = number of pantries currently owned by this color
        return float(np.sum(self.pantry_occ == owner_int))

    def _blue_wall_penalty(self) -> float:
        hit, _ = self._blue_wall_contact()
        return -0.02 if hit else 0.0

    def _score_both(self) -> tuple[float, float]:
        base_blue   = self._score_agent("blue")
        base_yellow = self._score_agent("yellow")
        
        pen_blue = self._blue_wall_penalty()
        return base_blue + pen_blue, base_yellow
    
    # -------------------- Simple scripted policy --------------------
    def _scripted_policy(self, agent: str) -> np.ndarray:
        p = self._body_xy(agent)
        th = self._body_yaw(agent)

        if self.carry[agent] < 0:
            # nearest available pickup
            candidates = [(i, name, PICKUPS[name]) for i, name in enumerate(self.pickup_names) if self.pickup_occ[i] == 1]
            tgt = min(candidates, key=lambda t: np.linalg.norm(t[2] - p))[2] if candidates else p
            op = 1 if within_circle(self._grip_xy(agent), tgt, PICKUP_R) else 0
        else:
            my_int = COLOR_TO_INT[agent]
            empty     = [(i, name, PANTRIES[name]) for i, name in enumerate(self.pantry_names) if self.pantry_occ[i] == 0]
            opp_owned = [(i, name, PANTRIES[name]) for i, name in enumerate(self.pantry_names) if self.pantry_occ[i] not in (0, my_int)]
            pool = empty if empty else opp_owned
            tgt = min(pool, key=lambda t: np.linalg.norm(t[2] - p))[2] if pool else p
            op = 2 if within_circle(p, tgt, PANTRY_R) else 0

        vec = tgt - p
        dist = float(np.linalg.norm(vec) + 1e-9)
        desired = float(np.arctan2(vec[1], vec[0]))
        ang_err = (desired - th + np.pi) % (2*np.pi) - np.pi

        # Turn-aggressive, move-conservative controller
        w_cmd = np.clip(1.8 * ang_err, -MAX_ANG_SPEED, MAX_ANG_SPEED)
        # reduce forward speed when misaligned or very close to target
        align_factor = np.clip(1.0 - abs(ang_err)/1.2, 0.1, 1.0)
        dist_factor  = np.clip(dist / 0.5, 0.0, 1.0)
        v_scale = float(np.clip(align_factor * dist_factor, 0.0, 1.0))

        return np.array([v_scale, w_cmd / MAX_ANG_SPEED, float(op)], dtype=np.float32)



