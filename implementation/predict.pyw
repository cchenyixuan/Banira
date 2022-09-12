import shutil
import sys
import os
import traceback

sys.path.extend([os.getcwd()])

from training_process.DiscrateClassifier import *
from multiprocessing import Pool
import numpy as np
import torch


def solver(buffer):
    model = TumourClassifier()
    model.load_state_dict(torch.load(sys.argv[1]))
    model = model.to("cuda")
    buffer = buffer.to("cuda")
    output = []
    for step, x in enumerate(buffer):
        x = x.reshape([1, 15, 10, 10])
        output.append(model(x).to("cpu").detach().numpy())
        if step % 100 == 0:
            print(step)
    return np.array(output, dtype=np.float32)


if __name__ == "__main__":
    tensors = [torch.from_numpy(np.load(sys.argv[2]))]
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
    os.makedirs(".//output_csv", exist_ok=True)
    os.makedirs(f"{os.path.dirname(os.path.dirname(sys.argv[1]))}//output_csv", exist_ok=True)
    with open(f".//output_csv//ai__{sys.argv[3]}.csv", "w") as f:
        for item, result in zip(torch.from_numpy(np.load(sys.argv[2])), answer):
            pos = (item[0, 0, 0], item[5, 0, 0], item[10, 0, 0])
            row = "{},{},{},{},{},{},{},{}\n".format(*pos, *result)
            f.write(row)
        f.close()
    shutil.copyfile(f".//output_csv//ai__{sys.argv[3]}.csv", f"{os.path.dirname(os.path.dirname(sys.argv[1]))}//output_csv//ai__{sys.argv[3]}.csv")
