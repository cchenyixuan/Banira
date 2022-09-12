import numpy as np
import sys
import csv
import re
import os
from multiprocessing import Pool


class Classifier:
    def __init__(self, column=6):
        sys.stdout.write("Accept type: *.csv, *.stl, *.npy or a folder containing these types.\n")
        sys.stdout.write("Vertex form: [x, y, z, value]\n")
        sys.stdout.write("             [x, y, z, r, g, b]\n")
        self.column = column

    def __call__(self, file_path):
        # when receiving a folder:
        if os.path.isdir(file_path):
            opt_list = [file_path + "/" + item for item in os.listdir(file_path)]
            pool = Pool()
            # data: [vertices, vertices, ...]
            data = pool.map(self.__call__, opt_list)
            buffer, pointer = self.pack_buffer(data, opt_list)
            return buffer, pointer

        # when receiving a file
        else:
            if file_path[-4:] == ".npy":
                return self.load_npy(file_path)
            elif file_path[-4:] == ".csv":
                if self.column == 2839:
                    return self.load_csv_2839(file_path)
                elif re.findall(re.compile(r"ai__", re.S), file_path):
                    return self.load_ai__(file_path)
                else:
                    return self.load_csv(file_path)
            elif file_path[-4:] == ".stl":
                return self.load_stl(file_path)
            else:
                sys.stderr.write(f"Unidentified file format: {file_path}!\n")

    def load_npy(self, file_path) -> np.ndarray:
        vertices = np.load(file_path)
        # [x, y, z]
        if vertices.shape[1] == 3:
            vertices = self.auto_gradient(vertices)
        # [x, y, z, value]
        elif vertices.shape[1] == 4:
            vertices = self.auto_gradient(vertices)
        # [x, y, z, r, g, b]
        elif vertices.shape[1] == 6:
            pass
        return vertices

    def load_csv(self, file_path):
        vertices = []
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for step, row in enumerate(csv_reader):
                if step == 0:
                    continue
                vertices.append([*row[:3], row[self.column]])
            f.close()
        vertices = np.array(vertices, dtype=np.float32)
        vertices = self.auto_gradient(vertices)
        return vertices

    def load_csv_2839(self, file_path):
        vertices = []
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for step, row in enumerate(csv_reader):
                if step == 0:
                    continue
                vertices.append([*row[:3], *row[3:6]])
            f.close()
        vertices = np.array(vertices, dtype=np.float32)
        return vertices

    def load_ai__(self, file_path):
        vertices = []
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for step, row in enumerate(csv_reader):
                if step == 0:
                    continue
                vertices.append([*row[:3], *row[3:8]])
            f.close()
        vertices = np.array(vertices, dtype=np.float32)
        return vertices

    @staticmethod
    def load_stl(file_path):
        find_vertex = re.compile(r"vertex (.+) (.+) (.+)\n", re.S)
        vertices = []
        with open(file_path, "r") as f:
            for row in f:
                vertex = re.findall(find_vertex, row)
                if vertex:
                    vertices.append([float(_) for _ in vertex[0]]+[0.3, 0.3, 0.3])
            f.close()
        vertices = np.array(vertices, dtype=np.float32)
        return vertices

    def auto_gradient(self, array):
        vertices = np.zeros([array.shape[0], 6], dtype=np.float32)
        sup = np.max(array[:, -1])
        inf = np.min(array[:, -1])
        for step, vertex in enumerate(array):
            x = 1-(vertex[-1]-inf) / (sup - inf)
            red, green, blue = self.color_gradient(x)
            vertices[step] = np.array([*vertex[:3], red, green, blue], dtype=np.float32)
        return vertices

    @staticmethod
    def color_gradient(x: float) -> tuple:
        red = max([0, -2.25*x**2+1])
        green = max([0, -4*x*(x-1)])
        blue = max([0, -2.25*(x-1)**2+1])
        return red, green, blue

    @staticmethod
    def pack_buffer(data: list, label: list):
        find_index = re.compile(r"(\d+)\....", re.S)
        sorted_data = [np.empty((0, 0))]*len(data)
        pointer = [[]] * len(data)
        for name, array in zip(label, data):
            sorted_data[int(re.findall(find_index, name)[0])] = array
        for i in range(len(data)):
            if i == 0:
                pointer[i] = [0, sorted_data[i].shape[0]]
            else:
                pointer[i] = [pointer[i-1][1], pointer[i-1][1] + sorted_data[i].shape[0]]
        buffer = np.concatenate(sorted_data, axis=0, dtype=np.float32)
        return buffer, pointer


if __name__ == '__main__':
    clf = Classifier()
    file_name = r"C:\PycharmProjects\stl\colorMap\CM_H_0.csv"
    a = clf(file_name)
    print(a.shape)
