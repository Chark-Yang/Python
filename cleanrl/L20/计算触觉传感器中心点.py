import numpy as np
import math

def calculate_tactile_sensor_pose(from_pt, to_pt, radius, ratio=0.8, pad_hint=[-1.0, 0.0, 0.0]):
    """
    计算灵巧手胶囊体表面的触觉传感器(Site)中心点坐标
    
    参数:
    from_pt: list/tuple, 胶囊体起点 (通常是 [0, 0, 0])
    to_pt: list/tuple, 胶囊体终点
    radius: float, 胶囊体的半径 (size)
    ratio: float, 传感器在骨骼全长上的位置比例 (0.0=根部, 1.0=指尖, 默认0.8靠近指尖)
    pad_hint: list/tuple, 指肚的大致朝向向量 (默认朝向负X方向)
    
    返回:
    dict: 包含表面绝对坐标 pos 和 参考法向量 normal
    """
    A = np.array(from_pt, dtype=float)
    B = np.array(to_pt, dtype=float)
    hint = np.array(pad_hint, dtype=float)
    
    # 1. 计算骨骼方向向量 V
    V = B - A
    bone_length = np.linalg.norm(V)
    if bone_length < 1e-6:
        raise ValueError("骨骼长度为0，请检查 fromto 坐标！")
    
    # 单位骨骼向量
    V_unit = V / bone_length
    
    # 2. 利用施密特正交化，寻找绝对垂直于骨骼的法向量
    # 公式：N = Hint - (Hint 点乘 V_unit) * V_unit
    projection_length = np.dot(hint, V_unit)
    N_raw = hint - projection_length * V_unit
    
    normal_length = np.linalg.norm(N_raw)
    if normal_length < 1e-6:
        raise ValueError("意图方向与骨骼方向完全平行，无法计算垂直面！请更换 pad_hint (例如换成 [0, -1, 0])")
        
    # 得到指向指肚的单位法向量
    N_unit = N_raw / normal_length
    
    # 3. 计算骨骼中心点 (基于比例)
    P_center = A + V * ratio
    
    # 4. 沿着法向量往外推一个半径的距离，得到表面坐标
    P_surface = P_center + N_unit * radius
    
    # 5. 扁平的椭圆形默认是平躺，需要绕Y轴旋转一个角度
    # 附加：粗略计算绕 Y 轴的 Pitch 旋转角（适用于只在 X-Z 平面弯曲的四根手指）
    # 以便让椭圆形的 site 能够顺着手指表面贴合
    pitch_angle = math.atan2(N_unit[0], N_unit[2])
    
    return {
        "pos": np.round(P_surface, 6).tolist(),
        "normal": np.round(N_unit, 4).tolist(),
        "euler_y_hint": round(pitch_angle, 4)
    }

# ==========================================
# 批量计算测试print("【大拇指 (Thumb Distal - 示例)】")
    # try:
    #     thumb_res = calculate_tactile_sensor_pose(
    #         from_pt=[0, 0, 0],
    #         to_pt=[0.005, -0.012, 0.025], # 假设的大拇指骨骼
    #         radius=0.006,                 # 假设大拇指稍微粗一点
    #         ratio=0.8,
    #         pad_hint=[-1.0, -1.0, 0.0]    # 告诉脚本大拇指指肚的粗略朝向
    #     )
    #     print(f"  建议 <site> pos:   {thumb_res['pos']}")
    #     print(f"  计算所得法向量:    {thumb_res['normal']}")
    # except Exception as e:
    #     print(f"计算出错: {e}")
# ==========================================
if __name__ == "__main__":
    
    print("=== 灵巧手触觉传感器坐标计算器 ===\n")

    # # 1. 测试食指末端 (Index Distal) - 验证之前的数学推导
    # index_res = calculate_tactile_sensor_pose(
    #     from_pt=[0, 0, 0],
    #     to_pt=[-0.011, 0, 0.01905],
    #     radius=0.0055,
    #     ratio=0.8,
    #     pad_hint=[1.0, 0.0, 0.0]  # 食指指肚朝内弯曲，大致朝向 X
    # )
    # print("【食指 (Index Distal)】")
    # print(f"  建议 <site> pos:   {index_res['pos']}")
    # print(f"  计算所得法向量:    {index_res['normal']}")
    # print(f"  建议 euler (俯仰): [0, {index_res['euler_y_hint']}, 0]\n")

    # 1. 测试中指末端 (Middle Distal) - 验证之前的数学推导
    middle_res = calculate_tactile_sensor_pose(
        from_pt=[0, 0, 0],
        to_pt=[-0.011, 0, 0.01905],
        radius=0.0055,
        ratio=0.8,
        pad_hint=[1.0, 0.0, 0.0]  # 食指指肚朝内弯曲，大致朝向 X
    )
    print("【中指 (Middle Distal)】")
    print(f"  建议 <site> pos:   {middle_res['pos']}")
    print(f"  计算所得法向量:    {middle_res['normal']}")
    print(f"  建议 euler (俯仰): [0, {middle_res['euler_y_hint']}, 0]\n")

    
    # 2. 假设你的大拇指末端参数 (请替换为你真实的 XML 数据)
    # 大拇指通常不仅在 X 轴弯曲，还会因为 Roll 关节在 Y 轴有偏移
    # 假设大拇指指肚朝向斜下方，比如大概在 -X 和 -Y 之间
    # print("【大拇指 (Thumb Distal - 示例)】")
    # try:
    #     thumb_res = calculate_tactile_sensor_pose(
    #         from_pt=[0, 0, 0],
    #         to_pt=[0.005, -0.012, 0.025], # 假设的大拇指骨骼
    #         radius=0.006,                 # 假设大拇指稍微粗一点
    #         ratio=0.8,
    #         pad_hint=[-1.0, -1.0, 0.0]    # 告诉脚本大拇指指肚的粗略朝向
    #     )
    #     print(f"  建议 <site> pos:   {thumb_res['pos']}")
    #     print(f"  计算所得法向量:    {thumb_res['normal']}")
    # except Exception as e:
    #     print(f"计算出错: {e}")