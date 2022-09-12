import numpy as np
import csv
import os

tensor = np.load(r"C:\Users\cchen\PycharmProjects\SEIKA_NUMERICAL_kumamoto/training/SEIKA_K08__xyz.npy")
label = np.load(r"C:\Users\cchen\PycharmProjects\SEIKA_NUMERICAL_kumamoto/training/SEIKA_K08__label.npy")
with open("included_ai__k08.csv", "w") as f:
    for item, index in zip(tensor,label):
        pos = (item[0, 0, 0], item[5, 0, 0], item[10, 0, 0])
        row = "{},{},{},{},{},{},{},{}\n".format(*pos, *index)
        f.write(row)
    f.close()
