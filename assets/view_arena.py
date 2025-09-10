import mujoco
from mujoco import viewer
m = mujoco.MjModel.from_xml_path("./assets/arena.xml")
d = mujoco.MjData(m)
with viewer.launch_passive(m, d) as v:
    for _ in range(400000):
        mujoco.mj_step(m, d)
        if not v.is_running():
            break
