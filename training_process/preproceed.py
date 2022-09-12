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
        if "zioformat" in os.listdir(self.file_path):
            self.files = [file for file in os.listdir(f"{self.file_path}/zioformat")]
        elif "predict" in os.listdir(self.file_path):
            self.files = [file for file in os.listdir(f"{self.file_path}/predict") if
                          self.error_csv_filter(f"{self.file_path}/predict/{file}")]
        else:
            print("Please check folder spelling.")

        self.quantite = len(self.files)
        self.buffer = np.zeros((1, 100, 4), dtype=np.float32)
        self.__call__()

    def __call__(self, *args, **kwargs):
        self.get_data()
        self.compute_gpgpu()

    @staticmethod
    def error_csv_filter(file):
        if os.path.getsize(file) > 2048:
            return True
        else:
            return False

    def csv_io(self, opt_list):
        buffer = np.zeros((len(opt_list), 100, 4), dtype=np.float32)
        sample_points = []
        for step, item in enumerate(opt_list):
            with open(f"{self.file_path}/predict/" + item, "r") as f:
                next(f)
                data = np.loadtxt(f, delimiter=',', dtype=np.float32)
                f.close()
            if step == 0:
                sample_points = [int(i * data.shape[0]/100) for i in range(100)]
            buffer[step] = np.array([data[i] for i in sample_points], dtype=np.float32)
        return buffer

    def zio_csv_io(self, opt_list):
        buffer = np.zeros((len(opt_list)*500, 100, 4), dtype=np.float32)
        dummy_time = np.array([0.01 * i for i in range(100)], dtype=np.float32).reshape((100, 1))
        sample_points = []
        quantite = 0
        for step, item in enumerate(opt_list):
            with open(f"{self.file_path}/zioformat/" + item, "r") as f:
                next(f)
                data = np.loadtxt(f, delimiter=',', dtype=np.float32)[:, 2:]
                f.close()
            if step == 0:
                sample_points = [int(i * data.shape[0]/100) for i in range(100)]
            for i in range(data.shape[1]//3):
                buffer[quantite] = np.hstack((dummy_time, np.array([data[:, i*3:i*3+3][j] for j in sample_points], dtype=np.float32)))
                quantite += 1
        return buffer[:quantite]

    def get_data(self):
        pool = Pool(60)
        batch_size = self.quantite//59
        arguments = [self.files[i*batch_size: (i+1)*batch_size] for i in range(59)] + [self.files[59*batch_size:]]
        if "zioformat" in os.listdir(self.file_path):
            self.buffer = np.vstack(pool.map(self.zio_csv_io, arguments))
        else:
            self.buffer = np.vstack(pool.map(self.csv_io, arguments))
        pool.close()
        self.quantite = self.buffer.shape[0]

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
        if os.path.exists(f"{self.root}\\banira_files\\npy\\{self.case_name}_xyz.npy"):
            np.save(f"{self.root}\\banira_files\\npy\\{self.case_name}_xyz_predict.npy",
                    np.vstack((output, np.load(f"{self.root}\\banira_files\\npy\\{self.case_name}_xyz.npy"))))
        else:
            np.save(f"{self.root}\\banira_files\\npy\\{self.case_name}_xyz_predict.npy", output)

        with open(f"{self.root}\\banira_files\\finished_ex.txt", "a") as f:
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

