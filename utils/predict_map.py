import numpy as np
import torch


with open("pdtSEIKA.csv", "r") as f:
    f_reader = np.loadtxt(f, delimiter=',', dtype=np.float32)
    predict = f_reader
    f.close()
tensor = torch.from_numpy(np.load(r"C:\Users\cchen\PycharmProjects\LearnPyTorch/K05_excluded_xyz.npy"))  # 101778,
# 15,10,10
with open("ai__K05_SEIKA2.csv", "w") as f:
    for item, result in zip(tensor, predict):
        pos = (item[0, 0, 0], item[5, 0, 0], item[10, 0, 0])
        row = "{},{},{},{},{},{},{},{}\n".format(*pos, *result)
        f.write(row)
    f.close()
