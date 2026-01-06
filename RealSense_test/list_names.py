"""
object_names:集合，利用集合的唯一性统计物体类别

object_counter:字典，统计每个物体的文件数量

grasp_code_list:用于获取文件夹中.npy文件总数

"""


import os
from collections import defaultdict

#路径
data_path = "/home/chark/DexGraspNet/data/dataset/dexgraspnet"

object_names = set()  # 用 set 自动去重
object_counter = defaultdict(int)


grasp_code_list = []
for code in os.listdir(data_path):
    # 只处理 .npy 文件;continue表示跳出当前循环，继续处理下一个文件
    if not code.endswith(".npy"):
        continue
    
    grasp_code_list.append(code[:-4])
    
    whole_name =code[:-4]
    parts = whole_name.split("-")
    if len(parts) >= 2:
        object_name = parts[1]          # core-[object]-hash
        object_counter[object_name] += 1

        # print(f"计数: {object_name} -> {object_counter[object_name]}")  # 调试计数
        object_names.add(object_name)

#排序
object_names = sorted(list(object_names))


print("文件夹中.npy文件总数：", len(grasp_code_list))

print("DexGraspNet 中的物体类别：")
print(object_names)
print("物体类别总数:", len(object_names))

print(sorted(object_counter.items()))
# for obj,count in sorted(object_counter.items()):
#     print(f"{obj}: {count} ")
print("object_counter的物体类别总数",len(object_counter))

# 检查 grasp_code_list 中包含 "_" 的文件名
# files_with_underscore = [name for name in grasp_code_list if "_" in name]
# if files_with_underscore:
#     print("包含 '_' 的文件名：", files_with_underscore)
# else:
#     print("grasp_code_list 中没有文件名包含 '_'")