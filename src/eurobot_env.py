import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
import mujoco
import mujoco.viewer

DT = 0.05  # 20 Hz ctrl

PANTRIES = {
    'A': np.array([ 0.8,  0.6]),
    'B': np.array([-0.8,  0.6]),
    'C': np.array([ 0.8, -0.6]),
    'D': np.array([-0.8, -0.6]),
}
NESTS = {'blue': np.array([-1.125, -0.7]), 'yellow': np.array([1.125, 0.7])}
PANTRY_R = 0.10
NEST_SIZE = np.array([0.225, 0.3])

def within_rect(p, center, half):
    d = np.abs(p - center)
    return (d[0] <= half[0]) and (d[1] <= half[1])

class EurobotMJ(ParallelEnv):
    metadata = {"name": "EurobotMJ-v0", "render_fps": int(1/DT)}
    def __init__(self, xml_path="assets/arena.xml", n_crates=8, max_steps=2000, scripted_opponent=True):
        self.xml_path = xml_path
        self.n_crates = n_crates
        self.max_steps = max_steps
        self.scripted_opponent = scripted_opponent
        self.agents = ['blue', 'yellow']
        self.pos_act = ['blue_x_act','blue_y_act','blue_yaw_act','yellow_x_act','yellow_y_act','yellow_yaw_act']
        self._build_spaces()
        self._model = mujoco.MjModel.from_xml_path(self.xml_path)
        self._data = mujoco.MjData(self._model)
        self._t = 0

        # carry state
        self.carry = {'blue': -1, 'yellow': -1}  # crate idx held
        self.rng = np.random.default_rng()

    def _build_spaces(self):
        # obs: ego (x,y,theta,carry), opp (x,y,theta), top-K crates (dx,dy,dist,free), pantry majority proxy (counts not simulated fully -> 0), time_left
        K = 6
        self.K = K
        obs_dim = 4 + 3 + K*4 + 1
        self.observation_spaces = {a: spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32) for a in self.agents}
        # action: [v, omega] in [-1,1], discrete op {0:noop,1:pick,2:drop}
        self.action_spaces = {a: spaces.Box(low=np.array([-1., -1., 0.]), high=np.array([1., 1., 2.]), dtype=np.float32) for a in self.agents}

    # ----- utility -----
    def _qpos_idx(self, name):
        j = self._model.joint(name).id
        return self._model.jnt_qposadr[j]

    def _body_pos(self, name):
        bid = self._model.body(name).id
        return self._data.xpos[bid][:2].copy()

    def _body_yaw(self, name):
        bid = self._model.body(name).id
        # yaw from xmat
        m = self._data.xmat[bid].reshape(3,3)
        yaw = np.arctan2(m[1,0], m[0,0])
        return yaw

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self._model, self._data)
        self._t = 0
        # spawn robots
        for a, pos in {'blue':(-1.0, -0.5), 'yellow':(1.0, 0.5)}.items():
            self._set_pose(a, np.array(pos), 0.0)
        # spawn crates jittered
        for i in range(self.n_crates):
            name = f"crate{i}"
            if i >= self._model.nbody:
                break
            p = self.rng.uniform(low=[-0.6,-0.4], high=[0.6,0.4])
            self._set_body2d(name, p, 0.0)
        self.carry = {'blue': -1, 'yellow': -1}
        self._step_count = 0
        obs = {a:self._obs(a) for a in self.agents}
        return obs, {a:{} for a in self.agents}

    def _set_pose(self, agent, pos, yaw):
        bx, by = f"{agent}_x", f"{agent}_y"
        self._data.qpos[self._qpos_idx(bx)] = pos[0]
        self._data.qpos[self._qpos_idx(by)] = pos[1]
        self._data.qpos[self._qpos_idx(f"{agent}_yaw")] = yaw

    def _set_body2d(self, name, pos, yaw):
        bid = self._model.body(name).id
        adr = self._model.body_jntadr[bid]
        self._data.qpos[self._model.jnt_qposadr[adr+0]] = pos[0]
        self._data.qpos[self._model.jnt_qposadr[adr+1]] = pos[1]
        self._data.qpos[self._model.jnt_qposadr[adr+2]] = yaw

    def _ego_tuple(self, a):
        p = self._body_pos(a)
        th = self._body_yaw(a)
        carry = 1.0 if self.carry[a] >= 0 else 0.0
        return p, th, carry

    def _obs(self, a):
        o = 1 if a=='blue' else 0
        me, opp = self.agents[o], self.agents[1-o]
        p, th, carry = self._ego_tuple(me)
        p_opp, th_opp, _ = self._ego_tuple(opp)

        # crates
        crates = []
        for i in range(self.n_crates):
            name = f"crate{i}"
            try:
                cp = self._body_pos(name)
            except Exception:
                continue
            free = 1.0 if (i != self.carry['blue'] and i != self.carry['yellow']) else 0.0
            d = cp - p
            crates.append(np.array([d[0], d[1], np.linalg.norm(d), free]))
        if len(crates)==0:
            crates = [np.zeros(4)]
        crates = np.array(crates)
        crates = crates[crates[:,2].argsort()][:self.K]
        if crates.shape[0] < self.K:
            crates = np.vstack([crates, np.zeros((self.K - crates.shape[0], 4))])

        time_left = 1.0 - (self._step_count / self.max_steps)
        obs = np.concatenate([p, [th, carry], p_opp, [th_opp], crates.flatten(), [time_left]]).astype(np.float32)
        return obs

    def step(self, actions):
        # opponent: scripted or last action if provided
        a_blue = actions['blue']
        if self.scripted_opponent:
            a_yellow = self._scripted('yellow')
        else:
            a_yellow = actions['yellow']

        # integrate simple kinematics via position actuators (target next pose)
        self._apply_agent('blue', a_blue)
        self._apply_agent('yellow', a_yellow)

        # step physics a few substeps
        sub = int(DT / self._model.opt.timestep)
        for _ in range(max(1, sub)):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        term = self._step_count >= self.max_steps
        obs = {a:self._obs(a) for a in self.agents}
        rew_b, rew_y = self._score()
        rews = {'blue': rew_b, 'yellow': rew_y}
        terms = {'blue': term, 'yellow': term}
        truncs = {'blue': False, 'yellow': False}
        infos = {'blue': {}, 'yellow': {}}
        return obs, rews, terms, truncs, infos

    def _apply_agent(self, agent, a):
        v = float(np.clip(a[0], -1, 1)) * 0.6  # m/s
        w = float(np.clip(a[1], -1, 1)) * 2.0  # rad/s
        op = int(round(np.clip(a[2], 0, 2)))

        p = self._body_pos(agent)
        th = self._body_yaw(agent)
        target = p + np.array([np.cos(th), np.sin(th)]) * v * DT
        yaw_t = th + w * DT

        self._data.ctrl[self._model.actuator('{}_x_act'.format(agent)).id] = target[0]
        self._data.ctrl[self._model.actuator('{}_y_act'.format(agent)).id] = target[1]
        self._data.ctrl[self._model.actuator('{}_yaw_act'.format(agent)).id] = yaw_t

        if op==1:  # pick nearest free crate within 0.18 m
            if self.carry[agent] == -1:
                idx = self._nearest_free_crate(agent, radius=0.18)
                if idx>=0: self.carry[agent] = idx
        elif op==2:  # drop
            if self.carry[agent] != -1:
                # place crate at grip pose
                gid = self.carry[agent]
                gpos = self._grip_pos(agent)
                self._set_body2d(f"crate{gid}", gpos, 0.0)
                self.carry[agent] = -1

    def _grip_pos(self, agent):
        site = self._model.site(f"{agent}_grip").id
        return self._data.site_xpos[site][:2].copy()

    def _nearest_free_crate(self, agent, radius=0.18):
        me_p = self._body_pos(agent)
        best, bestd = -1, 1e9
        for i in range(self.n_crates):
            if i == self.carry['blue'] or i == self.carry['yellow']:
                continue
            cp = self._body_pos(f"crate{i}")
            d = np.linalg.norm(cp - me_p)
            if d < bestd and d <= radius:
                best, bestd = i, d
        return best

    # simplistic score: +1 for crate in own nest area flat, +1 for crate in pantry area (any)
    def _score(self):
        def crate_score(owner):
            s = 0.0
            for i in range(self.n_crates):
                cp = self._body_pos(f"crate{i}")
                if within_rect(cp, NESTS[owner], NEST_SIZE): s += 1.0
                for ctr in PANTRIES.values():
                    if np.linalg.norm(cp-ctr) <= PANTRY_R: s += 0.5
            return s
        return crate_score('blue'), crate_score('yellow')

    def _scripted(self, agent):
        # greedy: pick nearest crate, carry to own nest, drop, repeat
        p = self._body_pos(agent); th = self._body_yaw(agent)
        if self.carry[agent] == -1:
            # go to nearest free crate
            idx = self._nearest_free_crate(agent, radius=10.0)
            tgt = self._body_pos(f"crate{idx}") if idx>=0 else NESTS[agent]
            op = 1 if np.linalg.norm(tgt - p) < 0.16 else 0
        else:
            tgt = NESTS[agent]
            op = 2 if within_rect(p, tgt, NEST_SIZE) else 0
        # steer to tgt
        v = 0.6
        ang = np.arctan2((tgt-p)[1], (tgt-p)[0])
        w = np.clip((ang - th + np.pi)%(2*np.pi)-np.pi, -2.0, 2.0)
        return np.array([v/0.6, w/2.0, op], dtype=np.float32)

    # optional viewer
    def render(self):
        viewer = mujoco.viewer.launch_passive(self._model, self._data)
        while viewer.is_running():
            mujoco.mj_step(self._model, self._data)
            viewer.sync()
