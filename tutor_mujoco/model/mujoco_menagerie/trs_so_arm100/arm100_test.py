import mujoco.viewer
import os

def main():

    # model = mujoco.MjModel.from_xml_path('model/mujoco_menagerie/aloha/scene.xml')
    
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