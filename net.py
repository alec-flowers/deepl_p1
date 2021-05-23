import torch.nn as nn
from torch.nn import functional as F
import torch


class NeuralNet(nn.Module):
    # Fully connected neural network with arbitrary hidden layers
    def __init__(self, input_size, hidden_sizes, output_size=1,
                 batchnorm_bool=False,
                 dropout_bool=False):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.dps = nn.ModuleList(
            [nn.Dropout(p=0.5) for _ in range(len(self.layers))])
        self.relus = nn.ModuleList(
            [nn.ReLU() for _ in range(len(self.layers))])
        self.sigmoid = nn.Sigmoid()

        self.batchnorm_bool = batchnorm_bool
        self.dropout_bool = dropout_bool

    def forward(self, x):
        for i, (layer, relu, bn, dp) in enumerate(zip(self.layers,
                                                      self.relus,
                                                      self.bns,
                                                      self.dps)):
            x = layer(x)
            if i + 1 < len(self.layers):
                if self.batchnorm_bool:
                    x = bn(x)
                if self.dropout_bool:
                    x = dp(x)
                x = relu(x)
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


class NeuralNetCalssifier(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size, hidden_sizes,
                 num_labels=10,
                 batchnorm_classifer_bool=False,
                 dropout_classifier_bool=False):
        super(NeuralNetCalssifier, self).__init__()
        self.input_size = input_size
        sizes = [input_size] + hidden_sizes + [num_labels]
        self.layers_classifier = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.bns_classifier = nn.ModuleList(
            [nn.BatchNorm1d(out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.dps_classifier = nn.ModuleList(
            [nn.Dropout(p=0.5) for _ in range(len(self.layers_classifier))])
        self.relus_classifier = nn.ModuleList(
            [nn.ReLU() for _ in range(len(self.layers_classifier))])

        self.batchnorm_calssifier_bool = batchnorm_classifer_bool
        self.dropout_calssifier_bool = dropout_classifier_bool

    def forward(self, x1, x2):
        for i, (layer, relu, bn, dp) in enumerate(zip(self.layers_classifier,
                                                      self.relus_classifier,
                                                      self.bns_classifier,
                                                      self.dps_classifier)):
            x1 = layer(x1)
            x2 = layer(x2)
            if i + 1 < len(self.layers_classifier):
                if self.batchnorm_calssifier_bool:
                    x1 = bn(x1)
                    x2 = bn(x2)
                if self.dropout_calssifier_bool:
                    x1 = dp(x1)
                    x2 = dp(x2)
                x1 = relu(x1)
                x2 = relu(x2)
        return torch.cat((x1, x2), 1)


class NeuralNetComparer(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size, hidden_sizes,
                 num_labels=10, output_size=1,
                 batchnorm_comparer_bool=False,
                 dropout_comparer_bool=False):
        super(NeuralNetComparer, self).__init__()
        sizes = [2 * num_labels] + hidden_sizes + [output_size]
        self.layers_comparer = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.bns_comparer = nn.ModuleList(
            [nn.BatchNorm1d(out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.dps_comparer = nn.ModuleList(
            [nn.Dropout(p=0.5) for _ in range(len(self.layers_comparer))])
        self.relus_comparer = nn.ModuleList(
            [nn.ReLU() for _ in range(len(self.layers_comparer))])
        self.sigmoid = nn.Sigmoid()

        self.batchnorm_comparer_bool = batchnorm_comparer_bool
        self.dropout_comparer_bool = dropout_comparer_bool

    def forward(self, x):
        inp = x
        for i, (layer, relu, bn, dp) in enumerate(zip(self.layers_comparer,
                                                      self.relus_comparer,
                                                      self.bns_comparer,
                                                      self.dps_comparer)):
            x = layer(x)
            if i + 1 < len(self.layers_comparer):
                if self.batchnorm_comparer_bool:
                    x = bn(x)
                if self.dropout_comparer_bool:
                    x = dp(x)
                x = relu(x)
        out = self.sigmoid(x)
        return out, inp


class NeuralNetCalssifierComparer(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size,
                 hidden_sizes_classifier, hidden_sizes_comparer,
                 num_labels=10, output_size=1,
                 aux_loss=False,
                 batchnorm_classifer_bool=False,
                 dropout_classifier_bool=False,
                 batchnorm_comparer_bool=False,
                 dropout_comparer_bool=False):
        super(NeuralNetCalssifierComparer, self).__init__()
        self.input_size = input_size
        self.classifier = NeuralNetCalssifier(
            input_size,
            hidden_sizes_classifier,
            batchnorm_classifer_bool=batchnorm_classifer_bool,
            dropout_classifier_bool=dropout_classifier_bool)
        self.comparer = NeuralNetComparer(
            input_size,
            hidden_sizes_comparer,
            batchnorm_comparer_bool=batchnorm_comparer_bool,
            dropout_comparer_bool=dropout_comparer_bool)

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
                 aux_loss=False,
                 batchnorm_classifer_bool=False,
                 dropout_classifier_bool=False,
                 batchnorm_comparer_bool=False,
                 dropout_comparer_bool=False):
        super(NeuralNetCalssifierComparerAuxLoss, self).__init__()
        self.input_size = input_size
        self.classifier = NeuralNetCalssifier(
            input_size,
            hidden_sizes_classifier,
            batchnorm_classifer_bool=batchnorm_classifer_bool,
            dropout_classifier_bool=dropout_classifier_bool)
        self.comparer = NeuralNetComparer(
            input_size,
            hidden_sizes_comparer,
            batchnorm_comparer_bool=batchnorm_comparer_bool,
            dropout_comparer_bool=dropout_comparer_bool)

        self.aux_loss = aux_loss
        self.num_labels = num_labels

    def forward(self, x):
        # x : 2, 14*14
        tgts, labels = self.comparer(self.classifier(x[:, 0, ...],
                                                     x[:, 1, ...]))
        return tgts, labels[:, :self.num_labels], labels[:, self.num_labels:]
