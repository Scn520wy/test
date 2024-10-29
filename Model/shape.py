import numpy as np

file_path = '/home/aixi/Documents/code/competion/ctr_test/pred.npy'  # 替换为你的文件路径
try:
    data = np.load(file_path)
    print("文件格式正确，数据形状为:", data.shape)
except Exception as e:
    print("文件格式错误:", e)
