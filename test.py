import torch
from torch import nn
from runner import *
from net import NeuralNet, Net, NeuralNetCalssifierComparer

test_rounds = 1
input_size = 2 * 14 * 14
hidden_sizes = [600, 600, 200]
lr = 1e-5
epochs = 100
batch_size = 100

for i in range(test_rounds):
    model = NeuralNet(input_size, hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    base = MLPRunner(model, criterion, optimizer, epochs, batch_size, name='mlp', writer_bool=True)
    base.run()

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