from training_process.DiscrateClassifier import *
from multiprocessing import Pool


def solver(buffer):
    model = TumourClassifier()
    model.load_state_dict(torch.load("./TumourClassifierSEIKA"))
    output = []
    for step, x in enumerate(buffer):
        x = x.reshape([1, 15, 10, 10])
        output.append(model(x).detach().numpy())
        if step % 100 == 0:
            print(step)
    return np.array(output, dtype=np.float32)


import re
import traceback


class Console:
    def __init__(self):
        print("This is a Interactive Python Console.")
        print("\n")
        self.code = """"""
        self.times = 1
        pass

    def fetch_input(self):
        raw_code = []
        tab = 0
        line = 0
        while True:
            line += 1
            if line == 1:
                sentence = input("In[{}]:".format(self.times) + "    " * tab)
            else:
                sentence = input("." * len("In[{}]:".format(self.times)) + "    " * tab)
            raw_code.append("   " * tab + sentence + "\n")
            if sentence == "":
                tab -= 1
            if tab < 0:
                break
            # check tab
            try:
                if sentence[-1] == ":":
                    tab += 1
            except IndexError:
                pass
            try:
                if sentence[:6] == "return":
                    tab -= 1
            except IndexError:
                pass
            try:
                if sentence[:5] == "break":
                    tab -= 1
            except IndexError:
                pass
            try:
                if sentence[:8] == "continue":
                    tab -= 1
            except IndexError:
                pass
            try:
                if sentence[:4] == "pass":
                    tab -= 1
            except IndexError:
                pass
        for item in raw_code:
            self.code += item

    def run_in_global(self):
        find_equal = re.compile(r"(.*)=(.*)", re.S)
        code = self.code.split("\n")
        for row in code:
            try:
                name_1, name_2 = re.findall(find_equal, row)[0]
                assert name_1[-1] != "+"
                assert name_1[-1] != "-"
                assert name_1[-1] != "*"
                assert name_1[-1] != "/"
                assert name_1[-1] != "!"
                assert name_1[-1] != "="
                if name_1 != row.split("=")[0]:  # multiple "=" in sentence
                    name_1 = row.split("=")[0]
                self.code = "global {}\n".format(name_1) + self.code
            except IndexError:
                pass
            except AssertionError:
                pass

    def __call__(self):
        while True:
            self.fetch_input()
            if self.code == """\n""":
                print(self.code)
                self.times -= 1
            self.run_in_global()
            if self.code == """quit\n\n""" or self.code == """exit\n\n""":
                break
            try:
                ans = eval(self.code)
                if ans is not None:
                    print("Out[{}]:".format(self.times) + str(ans) + "\n")
            except:
                try:
                    exec(self.code)
                except:
                    traceback.print_exc()
            self.code = """"""
            self.times += 1


if __name__ == "__main__":
    tensors = [torch.from_numpy(np.load(r"C:\Users\cchen\PycharmProjects\LearnPyTorch/K05_excluded_xyz.npy"))]
    for tensor in tensors:
        for i in range(15):
            tensor[:, i, :, :] -= torch.min(tensor[:, i, :, :])
            tensor[:, i, :, :] /= torch.max(tensor[:, i, :, :])
        tensor[:] *= 2
        tensor[:] -= 1.0
    X = torch.vstack(tensors)
    X = np.array(X, dtype=np.float32)
    X = torch.from_numpy(X)
    minibatch = X.shape[0] // 60
    arguments = [X[i * minibatch: i * minibatch + minibatch] for i in range(59)]
    arguments.append(X[59 * minibatch:])
    pool = Pool(10)
    answer = pool.map(solver, arguments)
    answer = [np.reshape(answer[i], (answer[i].shape[0], 5)) for i in range(60)]
    answer = np.vstack(answer)
    try:
        import csv

        with open("pdtSEIKA.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            for row in answer:
                csv_writer.writerow(row[::-1])
            f.close()
    except:
        traceback.print_exc()
    print("Done")

    pool.close()
"""    console = Console()
    console()
"""