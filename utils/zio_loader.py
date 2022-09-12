import numpy as np
import os
import sys

# (n, 100, 4)
class Dummy:
    def __init__(self, file_path, root):
        self.root = root
        self.file_path = file_path
        self.case_name = os.path.basename(self.file_path)
        # self.files = os.listdir(f"{self.file_path}/exclude/unclassified")
        self.quantite = 1000000
        self.buffer = np.zeros((self.quantite, 100, 4), dtype=np.float32)

    def zio_csv_io(self, opt_list):
        buffer = np.zeros((len(opt_list)*500, 100, 4), dtype=np.float32)
        dummy_time = np.array([0.01 * i for i in range(100)], dtype=np.float32).reshape((100, 1))
        sample_points = []
        quantite = 0
        for step, item in enumerate(opt_list):
            with open(f"{self.file_path}/zio_format/" + item, "r") as f:
                next(f)
                data = np.loadtxt(f, delimiter=',', dtype=np.float32)[:, 2:]
                f.close()
            if step == 0:
                sample_points = [int(i * data.shape[0]/100) for i in range(100)]
            for i in range(data.shape[1]//3):
                buffer[quantite] = np.hstack((dummy_time, np.array([data[:, i*3:i*3+3][j] for j in sample_points], dtype=np.float32)))
                quantite += 1
        return buffer[:quantite]

# graph draft
epo = len(box) // 3
if epo > 50:
    x, y, z = np.array(x), np.array(y), np.array(z)
    gap = (epo - 20) // 30
    rest = (epo - 20) % 30
    indice = []
    indice.append(0)
    for i in range(rest):
        indice.append(indice[-1] + gap + 1)
    for i in range(30-rest-1):
        indice.append(indice[-1] + gap)
    for i in range(epo-20, epo):
        indice.append(indice[-1] + 1)
    x = x[indice]
    y = y[indice]
    z = z[indice]