import torch
from enum import Enum

class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


def nb_errors(output, actual):
    if output.size(1) > 1:
        predict = output.argmax(1)
    else:
        predict = output.round()
    error = sum(~predict.eq(actual)).item()
    return error

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
    print(f"TEST ACCURACIES for {name} MEAN: {mean_test_acc.item()}, STD: {std_test_acc.item()}")
    print("\n")


if __name__ == '__main__':
    wrong = nb_errors(torch.tensor([[1, 0], [0, 1], [0, 1]]), torch.zeros((3)))
