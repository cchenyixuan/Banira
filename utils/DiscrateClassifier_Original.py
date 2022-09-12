import os

import torch
from torch.nn import Sequential, ReLU, Conv2d, Conv3d, MaxPool2d, MaxPool3d, Linear, Tanh, Sigmoid, Softmax, Flatten, \
    CrossEntropyLoss, Module

import torch.optim as optim
import numpy as np
import random


class TumourClassifier(Module):
    def __init__(self):
        super().__init__()
        self.x_block_k3 = Sequential(
            Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1,),  # b*5*10*10 -> b*15*10*10
            ReLU(),  # b*15*10*10 -> b*15*10*10
            MaxPool2d(2, 2),  # b*15*10*10 -> b*15*5*5

            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,),  # b*15*5*5 -> b*32*5*5
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

    tensors = [torch.from_numpy(np.load(r"C:\Users\cchen\PycharmProjects\seika_numerical\\" + item)) for item in os.listdir(r"C:\Users\cchen\PycharmProjects\seika_numerical\\") if item[-7:] == "xyz.npy"]
    for tensor in tensors:
        for i in range(15):
            tensor[:, i, :, :] -= torch.min(tensor[:, i, :, :])
            tensor[:, i, :, :] /= torch.max(tensor[:, i, :, :])
        tensor[:] *= 2
        tensor[:] -= 1.0
    X = torch.vstack(tensors)
    X = np.array(X, dtype=np.float32)
    X = torch.from_numpy(X).to("cuda")
    Y = torch.vstack([torch.from_numpy(np.load(r"C:\Users\cchen\PycharmProjects\seika_numerical\\" + item)) for item in os.listdir(r"C:\Users\cchen\PycharmProjects\seika_numerical\\") if item[-9:] == "label.npy"])
    Y.to("cuda")
    net = TumourClassifier().to("cuda")
    criterion = CrossEntropyLoss().to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=0.05)
    min_loss = 1e8
    for epoch in range(100000*len(tensors)):

        seeds = random.sample(range(X.shape[0]), 128)
        u = torch.vstack([X[i].reshape([1, 15, 10, 10]) for i in seeds]).to("cuda")
        v = torch.vstack([Y[i] for i in seeds]).to("cuda")

        running_loss = 0.0

        # get the inputs; data is a list of [inputs, labels]


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(u)
        loss = criterion(outputs, v)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if epoch % 100 == 0:
            print("loss: {} progress: {}%".format(running_loss, epoch/(100000*len(tensors))*100))
            if running_loss < min_loss:
                torch.save(net.state_dict(), "./TumourClassifierSEIKA")
                min_loss = running_loss

    print('Finished Training')



# %%
def evaluate(data_set, label) -> float:
    correct = 0
    total = 0
    for item, tag in zip(data_set, label):
        item = item.reshape([1, 15, 10, 10])
        if torch.argmax(net(item)) == torch.argmax(tag):
            correct += 1
        else:
            pass
        total += 1
    return correct/total

    ...
