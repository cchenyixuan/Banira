# %%
from interpolation import cubic_curve_fitting_parallel, points_velocity, points_accelerated_velocity, \
    points_third_derivation, points_fourth_derivation
import numpy as np
import os
import re
import sys
import time


#%%
def npy_creator_ex(filepath, root):
    find_case_name = re.compile(r".*\\([^\\]+)$", re.S)
    case_name = re.findall(find_case_name, filepath)[0]
    os.listdir(f"{filepath}/exclude/unclassified")
    x = os.listdir(f"{filepath}/exclude/unclassified")
    quantite = len(x)
    set_xyz = np.zeros((quantite, 15, 10, 10), dtype=np.float64)
    m = 0
    for step, item in enumerate(x):
        with open(f"{filepath}/exclude/unclassified/" + item, "r") as f:
            next(f)
            f_reader = np.loadtxt(f, delimiter=',', dtype=np.float64)
            f.close()
        f_reader = np.array(f_reader, dtype=np.float64)
        array_x = f_reader[:, :2]
        array_y = np.vstack((f_reader[:, 0], f_reader[:, 2])).transpose()
        array_z = np.vstack((f_reader[:, 0], f_reader[:, 3])).transpose()
        gap = f_reader[-1, 0] / 100
        t = np.array([i * gap for i in range(100)], dtype=np.float64)
        array_x = np.array(cubic_curve_fitting_parallel(array_x, t), dtype=np.float64)
        array_y = np.array(cubic_curve_fitting_parallel(array_y, t), dtype=np.float64)
        array_z = np.array(cubic_curve_fitting_parallel(array_z, t), dtype=np.float64)
        set_xyz[m, 1, :, :] = points_velocity(array_x, gap).reshape(10, 10)
        set_xyz[m, 2, :, :] = points_accelerated_velocity(array_x, gap).reshape(10, 10)
        set_xyz[m, 3, :, :] = points_third_derivation(array_x, gap).reshape(10, 10)
        set_xyz[m, 4, :, :] = points_fourth_derivation(array_x, gap).reshape(10, 10)
        set_xyz[m, 0, :, :] = array_x.reshape(10, 10)
        set_xyz[m, 6, :, :] = points_velocity(array_y, gap).reshape(10, 10)
        set_xyz[m, 7, :, :] = points_accelerated_velocity(array_y, gap).reshape(10, 10)
        set_xyz[m, 8, :, :] = points_third_derivation(array_y, gap).reshape(10, 10)
        set_xyz[m, 9, :, :] = points_fourth_derivation(array_y, gap).reshape(10, 10)
        set_xyz[m, 5, :, :] = array_y.reshape(10, 10)
        set_xyz[m, 11, :, :] = points_velocity(array_z, gap).reshape(10, 10)
        set_xyz[m, 12, :, :] = points_accelerated_velocity(array_z, gap).reshape(10, 10)
        set_xyz[m, 13, :, :] = points_third_derivation(array_z, gap).reshape(10, 10)
        set_xyz[m, 14, :, :] = points_fourth_derivation(array_z, gap).reshape(10, 10)
        set_xyz[m, 10, :, :] = array_z.reshape(10, 10)
        m += 1
        if m % 100 == 0:
            print(m)
    np.save(f"{root}/{case_name}_excluded_xyz.npy", set_xyz)


# %%
if __name__ == "__main__":
    npy_creator_ex(sys.argv[1], sys.argv[2])
