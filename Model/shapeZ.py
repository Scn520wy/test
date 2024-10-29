import numpy as np

file_path = '/home/aixi/Documents/code/competion/ctr_test/data/uav/UAV2B_bone_2d.npz'  # 替换为你的文件路径
try:
    data = np.load(file_path)
    print("文件格式正确，包含的数组有:", data.files)
    for key in data.files:
        print(f"数组 '{key}' 的形状为:", data[key].shape)
except Exception as e:
    print("文件格式错误:", e)
