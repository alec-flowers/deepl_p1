import torch
from torch import nn
from torch.nn import functional as F

# debugging stuff
# old_repr = torch.Tensor.__repr__
# def tensor_info(tensor):
#     return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
# torch.Tensor.__repr__ = tensor_info

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


def train_model(model, train_input, train_target, optimizer, criterion, lr=1e-1, epochs=100, batch_size=100, standardize=True):
    if standardize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

    for e in range(epochs):
        for input, target in zip(train_input.split(batch_size), train_target.split(batch_size)):
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if e%50 == 0:
            print(f'Epoch: {e}  Loss: {loss:.04f}')


def complete_nb_errors(model, input_data, target_data, batch_size):
    error = 0
    for input, target in zip(input_data.split(batch_size), target_data.split(batch_size)):
        output = model(input)
        _, predict = torch.max(output, 1)
        for i in range(target.size(0)):
            if predict[i] != target[i]:
                error += 1
    return error