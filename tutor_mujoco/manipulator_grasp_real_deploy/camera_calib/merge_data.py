"""
用于合并2个文件夹中的数据
"""

import os
import shutil
from pathlib import Path

src_dirs = ['260626_data_good', '260626_data_test']
dst_dir = 'merged'
os.makedirs(dst_dir, exist_ok=True)

idx = 0
for d in src_dirs:
    # 获取所有 jpg 文件并排序
    jpg_files = sorted(Path(d).glob('*.jpg'))
    for img_path in jpg_files:
        stem = img_path.stem
        pose_path = img_path.with_name(f"{stem}_pose.npy")
        if pose_path.exists():
            new_name = f"{idx:03d}"
            shutil.copy(img_path, f"{dst_dir}/{new_name}.jpg")
            shutil.copy(pose_path, f"{dst_dir}/{new_name}_pose.npy")
            print(f"Copied {img_path} -> {dst_dir}/{new_name}.jpg")
            idx += 1

print(f"Done! Total: {idx} groups.")