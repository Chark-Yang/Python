"""
此程序是jaka机械臂demo演示程序， 使用官方的urdf文件进行转换，然后又添加scene.xml,加入灯光及桌面等元素

运行该程序终端需要cd 到model/jaka_description目录下,这样第6行才能正确加载scene.xml文件


"""

import mujoco.viewer
import os

def main():

    model = mujoco.MjModel.from_xml_path('scene.xml')
    # 自动生成xml文件
    # mujoco.mj_saveLastXML("jaka_s5_auto.xml", model)
    data = mujoco.MjData(model)
 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
 
if __name__ == "__main__":
    main()