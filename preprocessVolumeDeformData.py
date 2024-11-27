import os
import shutil
import json

def preprocess_data(base_path):
    # 定义路径
    data_folder = os.path.join(base_path, "data")
    preprocessed_folder = os.path.join(base_path, "preprocessed")
    color_folder = os.path.join(preprocessed_folder, "color")
    depth_folder = os.path.join(preprocessed_folder, "depth")

    # 创建所需的文件夹
    os.makedirs(preprocessed_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    # 处理colorIntrinsics.txt
    intrinsics_path = os.path.join(data_folder, "colorIntrinsics.txt")
    new_intrinsics_path = os.path.join(preprocessed_folder, "intrinsics.txt")
    shutil.copyfile(intrinsics_path, new_intrinsics_path)

    # 遍历data文件夹，处理color.png和depth.png
    file_index = 0
    for file_name in sorted(os.listdir(data_folder)):
        if file_name.split(".")[1] == "color" and file_name.endswith(".png"):
            new_color_name = f"{file_index:04d}.png"
            shutil.copyfile(
                os.path.join(data_folder, file_name),
                os.path.join(color_folder, new_color_name),
            )
        elif file_name.split(".")[1] == "depth" and file_name.endswith(".png"):
            new_depth_name = f"{file_index:04d}.png"
            shutil.copyfile(
                os.path.join(data_folder, file_name),
                os.path.join(depth_folder, new_depth_name),
            )
            file_index += 1

# 使用base_path定义数据集的根路径
base_path = "data/shirt"
preprocess_data(base_path)
