import pyrealsense2 as rs       #用于控制realsense摄像头
import numpy as np              #处理图像数据
import cv2                          #用于显示图像

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)      #初始化RealSenseSense摄像头，并配置为捕获640x480的深度和颜色图像，每秒30帧
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:                                 #使用while循环不断捕获图像数据，直到用户关闭窗口。
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()            # 使用wait_for_frames()函数等待捕获到一组深度和颜色图像帧。
            depth_frame = frames.get_depth_frame()          #从帧中获取深度图像。
            color_frame = frames.get_color_frame()          #从帧中获取颜色图像。
            if not depth_frame or not color_frame:          #如果捕获到的帧中没有深度或颜色图像，则跳过当前循环，等待下一帧。
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())        # 将捕获到的深度图像转换为NumPy数组，以便进行后续处理。
            color_image = np.asanyarray(color_frame.get_data())        #将捕获到的颜色图像...

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)      #使用OpenCV库的applyColorMap()函数将深度图像转换为彩色图像，并使用cv2.hstack()函数将颜色和深度图像水平堆叠。
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)           #使用OpenCV库的namedWindow()函数创建一个窗口，并使用下面的imshow()函数将图像显示在窗口中。
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)                    #使用cv2.waitKey()函数等待用户按下键盘上的某个键，并返回按键的ASCII码。
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:             #如果用户按下'q'键或按ESC键，则使用cv2.destroyAllWindows()函数关闭窗口，并使用pipeline.stop()函数停止摄像头的流。
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()