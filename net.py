import torch.nn as nn
from torch.nn import functional as F
import torch


class NeuralNet(nn.Module):
    # Fully connected neural network with arbitrary hidden layers
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.relu(x)
        out = self.sigmoid(x)
        return out


class Net(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(576, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        shape0 = x.size()
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=1))
        shape1 = x.size()
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=1))
        shape2 = x.size()
        shape2h = x.view(-1, 576).size()
        x = F.relu(self.fc1(x.view(-1, 576)))
        shape3 = x.size()
        x = self.fc2(x)
        shape4 = x.size()
        return x


class NeuralNetCalssifierComparer(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size, hidden_sizes,
                 hidden_size2, num_labels=10, output_size=1):
        super(NeuralNetCalssifierComparer, self).__init__()
        self.input_size = input_size
        sizes = [input_size] + hidden_sizes + [num_labels]
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        # self.net = ParallelModule(nn.Sequential(self.layers, nn.Softmax()),
        #                           nn.Sequential(self.layers, nn.Softmax()))
        sizes2 = [2 * num_labels] + hidden_size2 + [output_size]
        self.layers2 = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes2, sizes2[1:])])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def classify(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.relu(x)
        out = self.softmax(x)
        return out

    def compare(self, x):
        for i, layer in enumerate(self.layers2):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.relu(x)
        out = self.sigmoid(x)
        return out

    def forward(self, x):
        # x : 2, 14*14
        labels1 = self.classify(x[:, 0, ...])
        labels2 = self.classify(x[:, 1, ...])
        labels = torch.cat((labels1,
                            labels2), 1)
        out = self.compare(labels)
        return out