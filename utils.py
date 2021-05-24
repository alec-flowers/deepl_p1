import torch
from enum import Enum

class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

class Verbosity(Enum):
    Full=2
    Some=1
    No=0



def report_from(run_output, name):
    rounds = len(run_output)
    train_loss = torch.zeros(rounds)
    test_loss = torch.zeros(rounds)
    train_accs = torch.zeros(rounds)
    test_accs = torch.zeros(rounds)
    for i in range(rounds):
        train_loss[i], test_loss[i],\
            train_accs[i], test_accs[i] =\
                (run_output[i])
    std_test_acc, mean_test_acc = torch.std_mean(test_accs)
    std_train_acc, mean_train_acc = torch.std_mean(train_accs)
    if rounds > 1:
        print(f"TRAIN ACCURACIES for {name} MEAN: {mean_train_acc.item():.04f}, STD: {std_train_acc.item():.04f}")
        print(f"TEST ACCURACIES for {name} MEAN: {mean_test_acc.item():.04f}, STD: {std_test_acc.item():.04f}")
    else:
        print(f"TRAIN ACCURACY for {name}: {mean_train_acc.item():.04f}")
        print(f"TEST ACCURACY for {name}: {mean_test_acc.item():.04f}")
    print("\n")
