import torch.nn as nn
import torch

##############################################################################
#######################   VANILA PLAIN NETWORKS ##############################
##############################################################################

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1,
                 batchnorm_bool=False,
                 dropout_bool=False):
        """
        Constructor for a NN (Vanilla MLP) used for classifying and comparison at the same time

        :param input_size:                  The size of the input for forward parse
        :param hidden_sizes:                List of sizes of hidden fully connected layers
        :param output_size:                 The size of the output of the NN
        :param batchnorm_bool :             Boolean determining whether \acitivating Batch normalization or not
        :param dropout_bool :               Boolean determining whether acitivating dropout or not
        """
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
        """
        :param x: The input of the NN 2*14*14
        """
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


class ConvNet_LeCun(nn.Module):
    '''
    Conv Net inspired by Lecun 1998 paper and design.
    Uses structure of first applying convolutions for feature extraction then running through fully
    connected layers for classification.
    '''
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


class ConvNet_LeCun_Update(nn.Module):
    '''
    Added BatchNormalization and dropout to the Le_Cun network to try and fix overfitting.
    '''
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
            # nn.Dropout(p=.5),
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


class ConvNet_VGG(nn.Module):
    '''
    Conv net inspired by VGG with two back to back Convolutions followed by BatchNorm, MaxPooling, and dropout. This is
    the network used in our paper.
    '''
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
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=.25)
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(p=.5),
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

##############################################################################
###########################  Specialized Classifier Networks  ################
##############################################################################


class NeuralNetCalssifier(nn.Module):
    def __init__(self, input_size, hidden_sizes,
                 num_labels=10,
                 batchnorm_classifer_bool=False,
                 dropout_classifier_bool=False):
        """
        Constructor for a NN working as the classifier sub-module

        :param input_size:                  The size of the input for forward parse
        :param hidden_sizes:                List of sizes of hidden fully connected layers for classifier sub-module
        :param num_labels:                  The size of the output of classifier sub-module
        :param batchnorm_classifer_bool :   Boolean determining whether acitivating Batch normalization or not for classifier sub-module
        :param dropout_classifer_bool :     Boolean determining whether acitivating dropout or not for classifier sub-module
        """
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
        '''
        :param x1:      First image
        :param x2:      Second image
        :return:        Concatenated output of first then second image
        '''
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


class CNNClassifier(nn.Module):
    '''
    Constructor for a CNN working as the classifier sub-module. Same structure as ConvNet_VGG but with softmax instead
    of sigmoid because we are using this for multi-class classification.
    '''
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
            nn.Dropout(p=.5),
            nn.Linear(576, 288),
            nn.BatchNorm1d(288),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.25),
            nn.Linear(288, 144),
            nn.BatchNorm1d(144),
            nn.ReLU(inplace=True),
            nn.Linear(144, 10),
            nn.Softmax(dim=0)
        )

    def forward(self, x1, x2):
        '''
        :param x1:      First Image
        :param x2:      Second Image
        :return:        Concatenated output of first then second image
        '''
        x1 = self.features(x1.unsqueeze(dim=1))
        x2 = self.features(x2.unsqueeze(dim=1))
        x1 = x1.view(-1, 576)
        x2 = x2.view(-1, 576)
        x1 = self.classifier(x1)
        x2 = self.classifier(x2)
        return torch.cat((x1, x2), 1)


##############################################################################
######################  Specialized Comparer Networks  #######################
##############################################################################

class NeuralNetComparer(nn.Module):
    def __init__(self, input_size, hidden_sizes,
                 num_labels=10, output_size=1,
                 batchnorm_comparer_bool=False,
                 dropout_comparer_bool=False):
        """
        Constructor for a NN used a sthe comparer sub-module

        :param input_size:                  The size of the input for forward parse
        :param hidden_sizes:                List of sizes of hidden fully connected layers
        :param num_labels:                  The size of the output of classifier sub-module
        :param batchnorm_comparer_bool :    Boolean determining whether acitivating Batch normalization or not for comparer sub-module
        :param dropout_comparer_bool :      Boolean determining whether acitivating dropout or not for comparer sublayer
        """
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
        '''
        :param x:   Input from the classifier sub-networks. Used to then run through this network and make final
                    prediction.
        :return:    Final prediction as probability.
        '''
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


##############################################################################
###########################  Classifier-Comparer Networks ####################
##############################################################################


class NeuralNetCalssifierComparer(nn.Module):
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size,
                 hidden_sizes_classifier,
                 hidden_sizes_comparer,
                 batchnorm_classifer_bool=False,
                 dropout_classifier_bool=False,
                 batchnorm_comparer_bool=False,
                 dropout_comparer_bool=False):
        """
        Constructor for a NN with two sub Module
        1. Classifier 2.Comparer

        :param input_size:                  The size of the input for forward parse
        :param hidden_sizes_classifier:     List of sizes of hidden fully connected layers for classifier sub-module
        :param hidden_sizes_comparer:       List of sizes of hidden fully connected layers for comparer sub-module
        :param batchnorm_classifer_bool :   Boolean determining whether \acitivating Batch normalization or not for classifier sub-module
        :param dropout_classifer_bool :     Boolean determining whether acitivating dropout or not for classifier sub-module
        :param batchnorm_comparer_bool :    Boolean determining whether acitivating Batch normalization or not for comparer sub-module
        :param dropout_comparer_bool :      Boolean determining whether acitivating dropout or not for comparer sublayer

        """
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

    def forward(self, x):
        '''
        :param x:       Two images of size 2x14x14
        :return:        Prediction
        '''
        tgts, _ = self.comparer(self.classifier(x[:, 0, ...],
                                                x[:, 1, ...]))
        return tgts


class CNNCalssifierComparer(nn.Module):
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size,
                 hidden_sizes_comparer,
                 batchnorm_comparer_bool=False,
                 dropout_comparer_bool=False):
        """
        Constructor for a CNN with two sub Module
        1. Classifier 2.Comparer

        :param input_size:                  The size of the input for forward parse
        :param hidden_sizes_comparer:       List of sizes of hidden fully connected layers for comparer sub-module
        :param batchnorm_comparer_bool :    Boolean determining whether acitivating Batch normalization or not for comparer sub-module
        :param dropout_comparer_bool :      Boolean determining whether acitivating dropout or not for comparer sublayer

        """
        super(CNNCalssifierComparer, self).__init__()
        self.input_size = input_size
        self.classifier = CNNClassifier()
        self.comparer = NeuralNetComparer(
            input_size,
            hidden_sizes_comparer,
            batchnorm_comparer_bool=batchnorm_comparer_bool,
            dropout_comparer_bool=dropout_comparer_bool)

    def forward(self, x):
        '''
        :param x:       Two images of size 2x14x14
        :return:        Prediction
        '''
        tgts, _ = self.comparer(self.classifier(x[:, 0, ...],
                                                x[:, 1, ...]))
        return tgts


##############################################################################
############ Classifier-Comaprer With Auxiliary Loss Networks ################
##############################################################################

class NeuralNetCalssifierComparerAuxLoss(nn.Module):
    # With two submodules: 1. classifier 2. comparer with auxiliary loss
    def __init__(self, input_size,
                 hidden_sizes_classifier,
                 hidden_sizes_comparer,
                 num_labels=10,
                 batchnorm_classifer_bool=False,
                 dropout_classifier_bool=False,
                 batchnorm_comparer_bool=False,
                 dropout_comparer_bool=False):
        """
        Constructor for a NN with two sub Module
        1. Classifier 2.Comparer

        :param input_size:                  The size of the input for forward parse
        :param hidden_sizes_classifier:                List of sizes of hidden fully connected layers for classifier sub-module
        :param hidden_sizes_comparer:       List of sizes of hidden fully connected layers for comparer sub-module
        :param num_labels:                  The size of the output of classifier sub-module
        :param batchnorm_classifer_bool :   Boolean determining whether \acitivating Batch normalization or not for classifier sub-module
        :param dropout_classifer_bool :     Boolean determining whether acitivating dropout or not for classifier sub-module
        :param batchnorm_comparer_bool :    Boolean determining whether acitivating Batch normalization or not for comparer sub-module
        :param dropout_comparer_bool :      Boolean determining whether acitivating dropout or not for comparer sublayer

        """
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

        self.num_labels = num_labels

    def forward(self, x):
        '''
        :param x:       Two images of size 2x14x14
        :return:        Prediction of boolean, label1, label2
        '''
        tgts, labels = self.comparer(self.classifier(x[:, 0, ...],
                                                     x[:, 1, ...]))
        return tgts, labels[:, :self.num_labels], labels[:, self.num_labels:]


class CNNCalssifierComparerAuxLoss(nn.Module):
    # With two submodules: 1. classifier 2. comparer with auxiliary loss
    def __init__(self, input_size,
                 hidden_sizes_comparer,
                 num_labels=10,
                 batchnorm_comparer_bool=False,
                 dropout_comparer_bool=False):
        """
        Constructor for a NN with two sub Module
        1. Classifier 2.Comparer

        :param input_size:                  The size of the input for forward parse
        :param hidden_sizes_comparer:       List of sizes of hidden fully connected layers for comparer sub-module
        :param num_labels:                  The size of the output of classifier sub-module
        :param batchnorm_comparer_bool :    Boolean determining whether acitivating Batch normalization or not for comparer sub-module
        :param dropout_comparer_bool :      Boolean determining whether acitivating dropout or not for comparer sublayer

        """
        super(CNNCalssifierComparerAuxLoss, self).__init__()
        self.input_size = input_size
        self.classifier = CNNClassifier()
        self.comparer = NeuralNetComparer(
            input_size,
            hidden_sizes_comparer,
            batchnorm_comparer_bool=batchnorm_comparer_bool,
            dropout_comparer_bool=dropout_comparer_bool)

        self.num_labels = num_labels

    def forward(self, x):
        '''
        :param x:       Two images of size 2x14x14
        :return:        Prediction of boolean, label1, label2
        '''
        tgts, labels = self.comparer(self.classifier(x[:, 0, ...],
                                                     x[:, 1, ...]))
        return tgts, labels[:, :self.num_labels], labels[:, self.num_labels:]
