import torch.nn as nn
from torch.nn import functional as F
import torch

old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info

class NeuralNet(nn.Module):
    # Fully connected neural network with arbitrary hidden layers
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.relus = nn.ModuleList(
            [nn.ReLU() for _ in range(len(self.layers))])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, (layer, relu) in enumerate(zip(self.layers, self.relus)):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = relu(x)
        out = self.sigmoid(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(576, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=1))
        x = F.relu(self.fc1(x.view(-1, 576)))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 17152)
        x = self.classifier(x)
        return x

class ConvNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(p=.25),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(p=.25)
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(p=.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.25),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

class ConvNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=.25),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=1),
            nn.Dropout(p=.25)
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(p=.5),
            nn.Linear(576, 288),
            nn.BatchNorm1d(288),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.25),
            nn.Linear(288, 144),
            nn.BatchNorm1d(144),
            nn.ReLU(inplace=True),
            nn.Linear(144, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 576)
        x = self.classifier(x)
        return x


class NeuralNetCalssifier(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size, hidden_sizes,
                 num_labels=10):
        super(NeuralNetCalssifier, self).__init__()
        self.input_size = input_size
        sizes = [input_size] + hidden_sizes + [num_labels]
        self.layers_classifier = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.relus_classifier = nn.ModuleList(
            [nn.ReLU() for _ in range(len(self.layers_classifier))])

    def forward(self, x1, x2):
        for i, (layer, relu) in enumerate(zip(self.layers_classifier,
                                              self.relus_classifier)):
            x1 = layer(x1)
            x2 = layer(x2)
            if i + 1 < len(self.layers_classifier):
                x1 = relu(x1)
                x2 = relu(x2)
        return torch.cat((x1, x2), 1)

class CNNClassifier(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=.25),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=.25)
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(p=.5),
            nn.Linear(576, 288),
            nn.BatchNorm1d(288),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.25),
            nn.Linear(288, 144),
            nn.BatchNorm1d(144),
            nn.ReLU(inplace=True),
            nn.Linear(144, 10),
            nn.Softmax()
        )

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)
        x1 = self.classifier(x1)
        x2 = self.classifier(x2)
        return torch.cat((x1, x2), 1)


class NeuralNetComparer(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size, hidden_sizes,
                 num_labels=10, output_size=1):
        super(NeuralNetComparer, self).__init__()
        sizes = [2 * num_labels] + hidden_sizes + [output_size]
        self.layers_comparer = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.relus_comparer = nn.ModuleList(
            [nn.ReLU() for _ in range(len(self.layers_comparer))])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inp = x
        for i, (layer, relu) in enumerate(zip(self.layers_comparer,
                                              self.relus_comparer)):
            x = layer(x)
            if i + 1 < len(self.layers_comparer):
                x = relu(x)
        out = self.sigmoid(x)
        return out, inp


class NeuralNetCalssifierComparer(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size,
                 hidden_sizes_classifier, hidden_sizes_comparer,
                 num_labels=10, output_size=1,
                 aux_loss=False):
        super(NeuralNetCalssifierComparer, self).__init__()
        self.input_size = input_size
        self.classifier = CNNClassifier()
        self.comparer = NeuralNetComparer(input_size,
                                          hidden_sizes_comparer)

        self.aux_loss = aux_loss

    def forward(self, x):
        # x : 2, 14*14
        tgts, _ = self.comparer(self.classifier(x[:, 0, ...],
                                                x[:, 1, ...]))
        return tgts


class NeuralNetCalssifierComparerAuxLoss(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size,
                 hidden_sizes_classifier, hidden_sizes_comparer,
                 num_labels=10, output_size=1,
                 aux_loss=False):
        super(NeuralNetCalssifierComparerAuxLoss, self).__init__()
        self.input_size = input_size
        self.classifier = NeuralNetCalssifier(input_size,
                                              hidden_sizes_classifier)
        self.comparer = NeuralNetComparer(input_size,
                                          hidden_sizes_comparer)

        self.aux_loss = aux_loss
        self.num_labels = num_labels

    def forward(self, x):
        # x : 2, 14*14
        tgts, labels = self.comparer(self.classifier(x[:, 0, ...],
                                                     x[:, 1, ...]))
        return tgts, labels[:, :self.num_labels], labels[:, self.num_labels:]
