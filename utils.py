import torch


def nb_errors(output, actual):
    if output.size(1) > 1:
        predict = output.argmax(1)
    else:
        predict = output.round()
    error = sum(~predict.eq(actual)).item()
    return error


if __name__ == '__main__':
    wrong = nb_errors(torch.tensor([[1, 0], [0, 1], [0, 1]]), torch.zeros((3)))