"""
该程序运行效果不好，jaka机械臂的正常demo程序在model/jaka_description/jaka_test1.py中，使用了scene.xml文件，加入了灯光和桌面等元素，运行效果更好
"""

import mujoco.viewer
import os

def main():

    model = mujoco.MjModel.from_xml_path('model/jaka_description/urdf/jaka_s5_copy.urdf')
    data = mujoco.MjData(model)
 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
 
if __name__ == "__main__":
    main()