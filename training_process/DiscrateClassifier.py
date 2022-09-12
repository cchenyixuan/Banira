import os
import sys
import time
import torch
from torch.nn import Sequential, ReLU, Conv2d, Conv3d, MaxPool2d, MaxPool3d, Linear, Tanh, Sigmoid, Softmax, Flatten, \
    CrossEntropyLoss, Module

import torch.optim as optim
import numpy as np
import random
from threading import Thread


def time_writer():
    while True:
        try:
            with open(f"{path}/time.txt", "a") as f:
                f.write(f"{present}\n{rest}\n")
                f.close()
        except NameError:
            present_0 = round(time.time() - start1)
            rest_0 = 'nan'
            with open(f"{path}/time.txt", "a") as f:
                f.write(f"{present_0}\n{rest_0}\n")
                f.close()
        time.sleep(0.5)


class TumourClassifier(Module):
    def __init__(self):
        super().__init__()
        self.x_block_k3 = Sequential(
            Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1, ),  # b*5*10*10 -> b*15*10*10
            ReLU(),  # b*15*10*10 -> b*15*10*10
            MaxPool2d(2, 2),  # b*15*10*10 -> b*15*5*5

            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, ),  # b*15*5*5 -> b*32*5*5
            ReLU(),  # b*32*5*5 -> b*32*5*5
        )
        self.y_block_k3 = Sequential(
            Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1, ),  # b*5*10*10 -> b*15*10*10
            ReLU(),  # b*15*10*10 -> b*15*10*10
            MaxPool2d(2, 2),  # b*15*10*10 -> b*15*5*5

            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, ),  # b*15*5*5 -> b*32*5*5
            ReLU(),  # b*32*5*5 -> b*32*5*5
        )
        self.z_block_k3 = Sequential(
            Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1, ),  # b*5*10*10 -> b*15*10*10
            ReLU(),  # b*15*10*10 -> b*15*10*10
            MaxPool2d(2, 2),  # b*15*10*10 -> b*15*5*5

            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, ),  # b*15*5*5 -> b*32*5*5
            ReLU(),  # b*32*5*5 -> b*32*5*5
        )

        self.x_block_k1 = Sequential(
            Conv2d(in_channels=5, out_channels=8, kernel_size=1, stride=1, padding=0, ),  # b*5*10*10 -> b*8*10*10
            ReLU(),  # b*8*10*10 -> b*8*10*10
            MaxPool2d(2, 2),  # b*8*10*10 -> b*8*5*5
        )
        self.y_block_k1 = Sequential(
            Conv2d(in_channels=5, out_channels=8, kernel_size=1, stride=1, padding=0, ),  # b*5*10*10 -> b*8*10*10
            ReLU(),  # b*8*10*10 -> b*8*10*10
            MaxPool2d(2, 2),  # b*8*10*10 -> b*8*5*5
        )
        self.z_block_k1 = Sequential(
            Conv2d(in_channels=5, out_channels=8, kernel_size=1, stride=1, padding=0, ),  # b*5*10*10 -> b*8*10*10
            ReLU(),  # b*8*10*10 -> b*8*10*10
            MaxPool2d(2, 2),  # b*8*10*10 -> b*8*5*5
        )

        self.affine = Sequential(
            Flatten(),  # b*120*5*5 -> b*3000
            Linear(5400, 1024),  # b*3000 -> b*1024
            ReLU(),
            Linear(1024, 256),  # b*1024 -> b*256
            Tanh(),
            Linear(256, 32),  # b*256 -> b*32
            Tanh(),
            Linear(32, 5),  # b*32 -> b*5
            Sigmoid(),
        )

    def forward(self, tensor):
        x = torch.cat((self.x_block_k3(tensor[:, :5, :, :]), self.x_block_k1(tensor[:, :5, :, :])), dim=1)
        y = torch.cat((self.y_block_k3(tensor[:, 5:10, :, :]), self.y_block_k1(tensor[:, 5:10, :, :])), dim=1)
        z = torch.cat((self.z_block_k3(tensor[:, 10:, :, :]), self.z_block_k1(tensor[:, 10:, :, :])), dim=1)
        buffer = torch.cat((x, y, z), dim=1)
        buffer = self.affine(buffer)
        return buffer


if __name__ == "__main__":
    # todo insert Pause & Stop signal
    # sys.argv[1]: root  str
    # sys.argv[2]: epoch value  int(str)
    # sys.argv[3]: test set ratio <- only use for self to self  int(str)
    # sys.argv[4]: 0->self to self; 1->other to self; 2->all to self
    # sys.argv[6]: train cases
    # sys.argv[5]: test set case <- use for other to self & all to self
    model_signal = int(sys.argv[4])
    if model_signal == 0:
        model_name = f"{100 - int(sys.argv[3])}to{sys.argv[3]}"
    elif model_signal == 1:
        model_name = f"allto{sys.argv[5]}"
    else:
        model_name = f"otherto{sys.argv[5]}"

    start1 = time.time()
    path = f"{sys.argv[1]}\\banira_files"
    train_cases = sys.argv[6].split(",")
    epoch_value = int(sys.argv[2])
    t1 = Thread(target=time_writer)
    t1.setDaemon(True)
    t1.start()

    os.makedirs(f"{path}\\TumourClassifier", exist_ok=True)

    tensors = [torch.from_numpy(np.load(f"{path}\\npy\\{item}_xyz.npy")) for item in train_cases]
    for tensor in tensors:
        for i in range(15):
            tensor[:, i, :, :] -= torch.min(tensor[:, i, :, :])
            tensor[:, i, :, :] /= torch.max(tensor[:, i, :, :])
        tensor[:] *= 2
        tensor[:] -= 1.0
        X = torch.vstack(tensors)
        X = np.array(X, dtype=np.float32)
        X = torch.from_numpy(X)
        Y = torch.vstack([torch.from_numpy(np.load(f"{path}\\npy\\{item}_xyz_label.npy")) for item in train_cases])

    if model_signal == 0:
        # 0->self to self
        test_size = int(int(sys.argv[3])/100 * X.shape[0])
        train_size = X.shape[0] - test_size
        indices = torch.randperm(X.shape[0])
        train_set = X[indices[:train_size]]
        Y_train = Y[indices[:train_size]]
        test_set = X[indices[train_size:]]
        Y_test = Y[indices[train_size:]]

    else:
        train_set = X
        Y_train = Y
        tensor_test = torch.from_numpy(np.load(f"{path}\\npy\\{sys.argv[5]}_xyz.npy"))
        for i in range(15):
            tensor_test[:, i, :, :] -= torch.min(tensor_test[:, i, :, :])
            tensor_test[:, i, :, :] /= torch.max(tensor_test[:, i, :, :])
        tensor_test[:] *= 2
        tensor_test[:] -= 1.0
        test_set = tensor_test
        Y_test = torch.from_numpy(np.load(f"{path}\\npy\\{sys.argv[5]}_xyz_label.npy"))


    train_set = train_set.to("cuda")
    test_set = test_set.to("cuda")
    Y_train = Y_train.to("cuda")
    Y_test = Y_test.to("cuda")
    net = TumourClassifier().to("cuda")
    criterion = CrossEntropyLoss().to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.05)
    min_loss = 1e8
    total = epoch_value * (train_set.shape[0] // 128)
    running_loss_10batch = 0.0
    batch = 0
    epoch = 0
    best_epoch = 0
    cnt = 0
    start2 = time.time()
    for epoch in range(int(epoch_value)):
        if "stop" in os.listdir(path):
            epoch -= 1
            break
        seeds = random.sample(range(train_set.shape[0]), train_set.shape[0])
        running_loss_epoch = 0.0
        for batch in range(train_set.shape[0] // 128):
            u = torch.vstack([train_set[i].reshape([1, 15, 10, 10]) for i in seeds[batch * 128:batch * 128 + 128]]).to("cuda")
            v = torch.vstack([Y_train[i] for i in seeds[batch * 128:batch * 128 + 128]]).to("cuda")
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(u)
            loss = criterion(outputs, v)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss_10batch += loss.item()
            running_loss_current = loss.item()
            running_loss_epoch += loss.item()

            while "pause" in os.listdir(path):
                time.sleep(1)
            if "stop" in os.listdir(path):
                break

            cnt += 1
            if cnt % 10 == 0:
                running_loss_10batch /= 10
                with open(f"{path}/training.txt", "a") as f:
                    f.write(
                        "Loss: {} Progress: {}%\n".format(running_loss_10batch, cnt / epoch_value / (train_set.shape[0] // 128) * 100))
                    f.close()
                running_loss_10batch = 0.0

            now = time.time()
            present = int(now - start1)
            rest = int((now - start2) / cnt * total - (now - start2))

        running_loss_epoch = running_loss_epoch / (batch + 1)
        optimizer.zero_grad()
        outputs = net(test_set)
        loss = criterion(outputs, Y_test)
        with open(f"{path}/graph.txt", "a") as f:
            f.write("{}\n{}\n{}\n".format(epoch+1, running_loss_epoch, loss))
            f.close()
        if loss < min_loss:
            torch.save(net.state_dict(), f"{path}\\TumourClassifier\\TumourClassifier")
            best_epoch = epoch + 1
            min_loss = loss

    time_stamp = "{}{}{}_{}{}{}".format(*[str(_).zfill(2) for _ in time.localtime()[:-3]])
    torch.save(net.state_dict(), f"{path}\\TumourClassifier\\{model_name}_CurrentEpoch_{epoch+1}_{time_stamp}")
    os.rename(f"{path}\\TumourClassifier\\TumourClassifier", f"{path}\\TumourClassifier\\{model_name}_BestEpoch_{best_epoch}_{time_stamp}")
    if "stop" in os.listdir(path):
        with open(f"{path}/training.txt", "a") as f:
            f.write("Training Interrupted\n")
            f.close()
    else:
        with open(f"{path}/training.txt", "a") as f:
            f.write("Training Finished\n")
            f.close()
    with open(f"{path}/time.txt", "a") as f:
        f.write(f"{present}\n0\n")
        f.close()



