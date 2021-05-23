import os
import torch
from torch import nn
from runner import *
from net import NeuralNet, Net, NeuralNetCalssifierComparer,\
    NeuralNetCalssifierComparerAuxLoss
from utils import report_from, Verbosity
verbose = Verbosity.Some

tensorboard_output = False
if tensorboard_output:
    os.system('rm -rf ./runs &')
    os.system('tensorboard --logdir=runs &')

test_rounds = 10
input_size_cc = 14 * 14  # MLP_RUNNER2
input_size_mlp = 2 * 14 * 14  # MLP_RUNNER
hidden_sizes = [600, 600, 200]
lr = 1e-5
epochs = 200
batch_size = 100

MLP_run_output = []
for i in range(test_rounds):
    model_mlp = NeuralNet(input_size_mlp, hidden_sizes)  # plain MLP

    optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=lr)
    criterion_mlp = nn.BCELoss()

    MLP = MLPRunner(model_mlp, [criterion_mlp], optimizer_mlp,
                    epochs, batch_size,
                    name=f'MLP_VANILLA_{i}', weights=[1.0],
                    writer_bool=tensorboard_output, verbose=verbose)
    MLP_run_output.append(MLP.run())
report_from(MLP_run_output, "MLP_VANILLA")

CC_run_output = []
for i in range(test_rounds):
    model_cc = NeuralNetCalssifierComparer(input_size_cc,
                                           hidden_sizes,
                                           hidden_sizes_comparer=[80, 80, 20])
    optimizer_cc = torch.optim.Adam(model_cc.parameters(), lr=lr)
    criterion_cc = nn.BCELoss()

    CC = MLPClassifierComparerRunner(
        model_cc, [criterion_cc], optimizer_cc,
        epochs, batch_size,
        name=f'MLP_classifier_comparer_{i}',
        writer_bool=tensorboard_output, verbose=verbose)
    CC_run_output.append(CC.run())
report_from(CC_run_output, "MLP_classifier_comparer")

CC_aux_run_output = []
for i in range(test_rounds):
    model_cc_aux = NeuralNetCalssifierComparerAuxLoss(
        input_size_cc,
        hidden_sizes,
        hidden_sizes_comparer=[80, 80, 20])
    optimizer_cc_aux = torch.optim.Adam(model_cc_aux.parameters(), lr=lr)
    criterion_cc_aux_main = nn.BCELoss()
    criterion_cc_aux_aux = nn.CrossEntropyLoss()

    MLP_CC_aux = MLPClassifierComparerRunner(
        model_cc_aux, [criterion_cc_aux_main,
                       criterion_cc_aux_aux,
                       criterion_cc_aux_aux], optimizer_cc_aux,
        epochs, batch_size,
        name=f'MLP_classifier_comparer_auxiliary_{i}',
        writer_bool=tensorboard_output, verbose=verbose)
    CC_aux_run_output.append(MLP_CC_aux.run())
report_from(CC_aux_run_output, "MLP_classifier_comparer_auxiliary")
# for i in range(test_rounds):
#     model = Net(10)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#
#     base = ConvRunner(model, criterion, optimizer, None, epochs, batch_size)
#     base.run()

# test_rounds = 3
# input_size = 14 * 14
# hidden_sizes2 = [80, 80, 20]
# hidden_sizes = [600, 600, 200]
#
# for i in range(test_rounds):
#     model = NeuralNetCalssifierComparer(input_size, hidden_sizes, hidden_sizes2)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCELoss()
#
#     base = MLPClassifierComparerRunner(model, criterion,
# optimizer, None, epochs, batch_size)
#     base.run()
