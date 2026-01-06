#python << 'EOF'
import pyrealsense2 as rs
import sys
 
print("=== RealSense SDK 功能测试 ===")
# 测试1：检查核心类是否可用
try:
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"测试1通过：上下文创建成功，找到 {len(devices)} 个设备")
except Exception as e:
    print(f"测试1失败：{e}")
 
# 测试2：如果连接了相机，尝试获取简要信息
try:
    if len(devices) > 0:
        dev = devices[0]
        print(f"  设备名称：{dev.get_info(rs.camera_info.name)}")
        print(f"  序列号：{dev.get_info(rs.camera_info.serial_number)}")
except Exception as e:
    print(f"测试2（设备信息）跳过或无设备：{e}")
 
print("=== 测试完成 ===")