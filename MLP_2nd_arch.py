from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

from dlc_practical_prologue import *

data = generate_pair_sets(1000)
train_data = data[0]
train_labels = data[1]
train_classes = data[2]

test_data = data[3]
test_labels = data[4]
test_classes = data[5]

num_labels = 10
lr = 1.0e-4
epochs = 10
batch_size = 10
standardize = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


class CustomDataset(Dataset):
    def __init__(self, inputs, targets, classes):
        super(Dataset, self).__init__()
        self.data = []
        for inp, tgt, cla in zip(inputs, targets, classes):
            self.data.append([inp, [tgt, cla]])
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


train_dataset = CustomDataset(train_data, train_labels, train_classes)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = CustomDataset(test_data, test_labels, test_classes)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


input_size = 14 * 14
hidden_sizes2 = [80, 80, 20]
hidden_sizes = [600, 600, 200]


class NeuralNetCalssifierComparer(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size, hidden_sizes,
                 hiddens_size2, num_labels=10, output_size=1):
        super(NeuralNetCalssifierComparer, self).__init__()
        self.input_size = input_size
        sizes = [input_size] + hidden_sizes + [num_labels]
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        # self.net = ParallelModule(nn.Sequential(self.layers, nn.Softmax()),
        #                           nn.Sequential(self.layers, nn.Softmax()))
        sizes2 = [2 * num_labels] + hidden_sizes2 + [output_size]
        self.layers2 = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes2, sizes2[1:])])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def classify(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.relu(x)
        out = self.softmax(x)
        return out

    def compare(self, x):
        for i, layer in enumerate(self.layers2):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = self.relu(x)
        out = self.sigmoid(x)
        return out

    def forward(self, x):
        # x : 2, 14*14
        labels1 = self.classify(x[:, 0, ...])
        labels2 = self.classify(x[:, 1, ...])
        labels = torch.cat((labels1,
                            labels2), 1)
        out = self.compare(labels)
        return out


model_new = NeuralNetCalssifierComparer(
    input_size, hidden_sizes, hidden_sizes2)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model_new.parameters(), lr=lr)


n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (inps, [tgts, clas]) in enumerate(train_loader):

        # origin shape: [n, 2, 14, 14]
        # resized: [n, 2*14*14]
        inps = inps.reshape(-1, 2, 14*14).to(device).float()
        tgts = tgts.to(device).reshape(-1, 1).float()

        # Forward pass
        outputs = model_new(inps)
        loss = criterion(outputs, tgts)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}]"
                  + f", Loss: {loss.item():.4f}")


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (inps, [tgts, clas]) in enumerate(test_loader):
        inps = inps.reshape(-1, 2, 14*14).to(device).float()
        tgts = tgts.to(device).reshape(-1, 1).float()
        predicted = model_new(inps)
        predicted_cls = predicted.round()
        n_samples += tgts.size(0)
        n_correct += (predicted_cls == tgts).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
