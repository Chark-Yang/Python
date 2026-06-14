

import jkrc

def jaka_get_tcp_position():
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  
    print("获取机械臂末端位置") 

    ret = robot.get_tcp_position()  
    if ret[0] == 0:  
        print("the tcp position is :",ret[1])  
    else:  
        print("some things happend,the errcode is: ",ret[0])  
    
    robot.logout()
    
    return ret[1]


# =========================
# JAKA 配置
# =========================

# 机械臂初始化
robot = jkrc.RC("192.168.2.155")#返回机器人对象  
ss = jaka_get_tcp_position()