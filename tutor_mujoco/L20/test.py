import mujoco.viewer
import os
import time


URDF_PATH = os.path.join(os.path.dirname(__file__), 'right', 'linkerhand_l20_right.urdf')

XML_PATH = os.path.join(os.path.dirname(__file__), 'linker_hand.xml')

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # 仿真主循环
    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
