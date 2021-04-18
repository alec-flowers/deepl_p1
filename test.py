import torch
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue
from utils import train_model, complete_nb_errors, Net

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

test_rounds = 3
hidden = 50
lr = 1e-1
epochs = 100
batch_size = 100
standardize = True

error_list = []
for i in range(test_rounds):
    model = Net(hidden)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_input, train_target, optimizer, criterion, lr, epochs, batch_size, standardize)
    error = complete_nb_errors(model, test_input, test_target, batch_size)

    print(f"==== Round: {i+1} - Accuracy: {100*error/test_input.size(0):.02f}%  {error:d}/{test_input.size(0):d} ====")
    error_list.append(error)
print(f'\nAverage Error: {100*sum(error_list)/(test_input.size(0)*test_rounds):.02f}%')
