import os
import sys
import re
import pyrr
import glfw
import time
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from multiprocessing import Pool



class Preprocessor:
    def __init__(self, file_path, root):
        self.root = root
        self.file_path = file_path
        self.case_name = os.path.basename(self.file_path)
        self.folders = os.listdir(f"{self.file_path}/training/")

        find_blue = re.compile(r"blu", re.S)
        find_skyblue = re.compile(r"sky", re.S)
        find_green = re.compile(r"gre", re.S)
        find_yellow = re.compile(r"yel", re.S)
        find_red = re.compile(r"red", re.S)
        self.color = ["blue", "skyblue", "green", "yellow", "red"]

        self.file_path_dict = {
            "blue":
                [f"{self.file_path}/training/" + folder for folder in self.folders if
                 re.findall(find_blue, folder[:5])][0],
            "skyblue":
                [f"{self.file_path}/training/" + folder for folder in self.folders if
                 re.findall(find_skyblue, folder[:5])][0],
            "green":
                [f"{self.file_path}/training/" + folder for folder in self.folders if
                 re.findall(find_green, folder[:5])][0],
            "yellow":
                [f"{self.file_path}/training/" + folder for folder in self.folders if
                 re.findall(find_yellow, folder[:5])][0],
            "red":
                [f"{self.file_path}/training/" + folder for folder in self.folders if
                 re.findall(find_red, folder[:5])][0],

        }



        self.file_dict = {
            "blue": [self.file_path_dict["blue"]+"//"+file for file in os.listdir(self.file_path_dict["blue"]) if self.error_csv_filter(self.file_path_dict["blue"]+"//"+file, file, "blue")],
            "skyblue": [self.file_path_dict["skyblue"]+"//"+file for file in os.listdir(self.file_path_dict["skyblue"]) if self.error_csv_filter(self.file_path_dict["skyblue"]+"//"+file, file, "skyblue")],
            "green": [self.file_path_dict["green"]+"//"+file for file in os.listdir(self.file_path_dict["green"]) if self.error_csv_filter(self.file_path_dict["green"]+"//"+file, file, "green")],
            "yellow": [self.file_path_dict["yellow"]+"//"+file for file in os.listdir(self.file_path_dict["yellow"]) if self.error_csv_filter(self.file_path_dict["yellow"]+"//"+file, file, "yellow")],
            "red": [self.file_path_dict["red"]+"//"+file for file in os.listdir(self.file_path_dict["red"]) if self.error_csv_filter(self.file_path_dict["red"]+"//"+file, file, "red")],
        }

        self.files = [csv_file for color in self.color for csv_file in self.file_dict[color]]
        self.quantite = sum([len(label) for label in self.file_dict.values()])
        self.buffer = np.zeros((self.quantite, 100, 4), dtype=np.float32)
        self.label = np.zeros((self.quantite, 5), dtype=np.float32)
        self.__call__()

    def __call__(self, *args, **kwargs):
        self.get_data()
        self.compute_gpgpu()
        self.write_labels()

    def error_csv_filter(self, file, filename, color):
        if os.path.getsize(file) > 2048:
            return True
        else:
            with open(f"{self.root}/banira_files/log.txt", "a") as f:
                f.write(f"File error: {self.case_name} {color} {filename}\n")
                f.close()
            return False

    def write_labels(self):
        lengths = [len(self.file_dict[color]) for color in self.color]  # [1000, 1000, 1000, 1000, 1000]
        offsets = [sum(lengths[:i]) for i in range(len(lengths))] + [self.quantite]
        for i in range(len(lengths)):
            self.label[offsets[i]:offsets[i+1], i] = 1
        np.save(f"{self.root}\\banira_files\\npy\\{self.case_name}_xyz_label.npy", self.label)

    def csv_io(self, opt_list):
        buffer = np.zeros((len(opt_list), 100, 4), dtype=np.float32)
        sample_points = []
        for step, item in enumerate(opt_list):
            with open(item, "r") as f:
                next(f)
                data = np.loadtxt(f, delimiter=',', dtype=np.float32)
                f.close()
            if step == 0:
                sample_points = [int(i * data.shape[0]/100) for i in range(100)]
            buffer[step] = np.array([data[i] for i in sample_points], dtype=np.float32)
        with open(f"{self.root}/banira_files/counter.txt", "a") as g:
            g.write(f"{len(opt_list)}\n")
            g.close()
        return buffer

    def get_data(self):
        pool = Pool(36)
        batch_size = self.quantite//35
        arguments = [self.files[i*batch_size: (i+1)*batch_size] for i in range(35)] + [self.files[35*batch_size:]]
        self.buffer = np.vstack(pool.map(self.csv_io, arguments))
        pool.close()

    def compute_gpgpu(self):
        glfw.init()
        window = glfw.create_window(100, 100, "OpenGL", None, None)
        glfw.hide_window(window)
        glfw.make_context_current(window)
        compute_shader_src = open(f"{os.getcwd()}\\training_process\\preproceed_compute_shader.shader", "r").read()
        compute_shader = compileProgram(compileShader(compute_shader_src, GL_COMPUTE_SHADER))
        glUseProgram(compute_shader)

        self.compute_sbo1 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.compute_sbo1)
        buf1 = np.zeros((4 * 100 * self.quantite,), dtype=np.float32)
        glBufferData(GL_SHADER_STORAGE_BUFFER, buf1.nbytes, buf1, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.compute_sbo1)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.buffer.nbytes, self.buffer)

        self.compute_sbo2 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.compute_sbo2)
        buf2 = np.zeros((4 * 4 * 100 * self.quantite,), dtype=np.float32)
        glBufferData(GL_SHADER_STORAGE_BUFFER, buf2.nbytes, buf2, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.compute_sbo2)

        glUseProgram(compute_shader)
        glDispatchCompute(self.quantite, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self.ans = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.buffer.shape[0]*16*100*4), dtype=np.float32)
        # x0, x0', x0'', x0''', x0'''', y0, y0', y0'', y0''', y0'''', z0, z0', z0'', z0''', z0'''', 0, ..., x99, ..., z99'''', 0, ...*101987
        # np.save("ans.npy", self.ans)
        output = np.zeros((self.quantite, 15, 10, 10), dtype=np.float32)
        for i in range(self.quantite):
            tmp = self.ans[i*1600:(i+1)*1600].reshape((100, 4, 4))
            output[i][0] = tmp[:, 0, 0].reshape((10, 10))
            output[i][1] = tmp[:, 0, 1].reshape((10, 10))
            output[i][2] = tmp[:, 0, 2].reshape((10, 10))
            output[i][3] = tmp[:, 0, 3].reshape((10, 10))
            output[i][4] = tmp[:, 1, 0].reshape((10, 10))
            output[i][5] = tmp[:, 1, 1].reshape((10, 10))
            output[i][6] = tmp[:, 1, 2].reshape((10, 10))
            output[i][7] = tmp[:, 1, 3].reshape((10, 10))
            output[i][8] = tmp[:, 2, 0].reshape((10, 10))
            output[i][9] = tmp[:, 2, 1].reshape((10, 10))
            output[i][10] = tmp[:, 2, 2].reshape((10, 10))
            output[i][11] = tmp[:, 2, 3].reshape((10, 10))
            output[i][12] = tmp[:, 3, 0].reshape((10, 10))
            output[i][13] = tmp[:, 3, 1].reshape((10, 10))
            output[i][14] = tmp[:, 3, 2].reshape((10, 10))
        np.save(f"{self.root}\\banira_files\\npy\\{self.case_name}_xyz.npy", output)
        with open(f"{self.root}\\banira_files\\log.txt", "a") as f:
            f.write(f"{self.case_name} finished\n")
            f.close()
        with open(f"{self.root}\\banira_files\\finished.txt", "a") as f:
            f.write(f"{self.case_name}\n")
            f.close()
        glfw.terminate()




if __name__ == "__main__":
    os.makedirs(f"{sys.argv[1]}\\banira_files\\npy", exist_ok=True)
    cmd = ""
    for c in sys.argv[2:]:
        cmd += c
    for item in eval(cmd):
        Preprocessor(f"{sys.argv[1]}//{item}", sys.argv[1])

