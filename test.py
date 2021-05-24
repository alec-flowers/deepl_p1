import torch
from torch import nn
from runner import *
from net import *

test_rounds = 1
input_size_cc = 14 * 14  # MLP_RUNNER2
input_size_mlp = 2 * 14 * 14  # MLP_RUNNER
hidden_sizes = [600, 600, 200]
lr = 1e-5
epochs = 200
batch_size = 10

# for i in range(test_rounds):
#     model_mlp = ConvNet4()  # plain MLP
#
#     optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=lr)
#     criterion_mlp = nn.BCELoss()
#
#     base_mlp = ConvRunner(model_mlp, [criterion_mlp], optimizer_mlp,
#                          epochs, batch_size,
#                          name='CovNet4',writer_bool=True)
#     base_mlp.run()

# for i in range(test_rounds):
#     model_mlp = NeuralNet(input_size_mlp, hidden_sizes)  # plain MLP
#
#     optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=lr)
#     criterion_mlp = nn.BCELoss()
#
#     base_mlp = MLPRunner(model_mlp, [criterion_mlp], optimizer_mlp,
#                          epochs, batch_size,
#                          name='mlp', weights=[1.0] ,writer_bool=True)
#     base_mlp.run()

for i in range(test_rounds):
    model_cc = NeuralNetCalssifierComparer(input_size_cc,
                                           hidden_sizes,
                                           hidden_sizes_comparer=[80, 80, 20])
    optimizer_cc = torch.optim.Adam(model_cc.parameters(), lr=lr)
    criterion_cc = nn.BCELoss()

    base_cc = CNNRunner(model_cc, [criterion_cc], optimizer_cc,
                         epochs, batch_size,
                         name='CNN_classifier_comparer', writer_bool=True)
    base_cc.run()
#
#
# for i in range(test_rounds):
#     model_cc_aux = NeuralNetCalssifierComparerAuxLoss(
#         input_size_cc,
#         hidden_sizes,
#         hidden_sizes_comparer=[80, 80, 20])
#     optimizer_cc_aux = torch.optim.Adam(model_cc_aux.parameters(), lr=lr)
#     criterion_cc_aux_main = nn.BCELoss()
#     criterion_cc_aux_aux = nn.CrossEntropyLoss()
#
#     base_cc_aux = MLP2Runner(
#         model_cc_aux, [criterion_cc_aux_main,
#                        criterion_cc_aux_aux,
#                        criterion_cc_aux_aux], optimizer_cc_aux,
#         epochs, batch_size,
#         name='mlp_classifier_comparer_aux', writer_bool=True)
#     base_cc_aux.run()

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
#     base = MLP2Runner(model, criterion, optimizer, None, epochs, batch_size)
#     base.run()
