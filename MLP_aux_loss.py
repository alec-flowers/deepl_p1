from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

from dlc_practical_prologue import *

data_size = 5000
data = generate_pair_sets(data_size)
train_data = data[0]
train_labels = data[1]
train_classes = data[2]

test_data = data[3]
test_labels = data[4]
test_classes = data[5]

num_labels = 10
lr = 5.0e-6
epochs = 100
batch_size = 100
standardize = True

w_main = 0.5
w_aux = 0.25


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
hidden_sizes_compare = [20, 20]
hidden_sizes_classify = [600, 600, 200]


class NeuralNetCalssifierComparer(nn.Module):
    # Fully connected neural network with one hidden layer
    # With two submodules: 1. classifier 2. comparer
    def __init__(self, input_size, hidden_sizes_classify,
                 hiddens_sizes_compare, num_labels=10, output_size=1):
        super(NeuralNetCalssifierComparer, self).__init__()
        self.input_size = input_size
        sizes_classify = [input_size] + hidden_sizes_classify + [num_labels]
        self.layers_classify = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes_classify,
                                                           sizes_classify[1:])])
        # sizes_compare =/
        # [2 * num_labels] + hidden_sizes_compare + [output_size]
        sizes_compare = [2] + hidden_sizes_compare + [output_size]
        self.layers_compare = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes_compare,
                                                           sizes_compare[1:])])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def classify(self, x):
        for i, layer in enumerate(self.layers_classify):
            x = layer(x)
            if i + 1 < len(self.layers_classify):
                x = self.relu(x)
        out = x
        return out

    def compare(self, x):
        # x : n, 2
        for i, layer in enumerate(self.layers_compare):
            x = layer(x)
            if i + 1 < len(self.layers_compare):
                x = self.relu(x)
        out = self.sigmoid(x)
        return out

    def forward(self, x):
        # x : n, 2, 14, 14
        labels1 = self.classify(x[:, 0, ...])
        labels2 = self.classify(x[:, 1, ...])
        labels = torch.cat((labels1,
                            labels2), 1)
        labels1_softmax = self.softmax(labels1)
        labels2_softmax = self.softmax(labels2)
        _, predicted1 = torch.max(labels1_softmax.data, 1)
        _, predicted2 = torch.max(labels2_softmax.data, 1)
        predicted1 = predicted1.reshape([-1, 1]).float()
        predicted2 = predicted2.reshape([-1, 1]).float()
        # print(f"{predicted1=}")
        # print(f"{predicted2=}")
        predicts = torch.cat((predicted1,
                              predicted2), 1)
        # print(f"{predicts=}")
        comparison_out = self.compare(predicts)
        return comparison_out, labels1, labels2


model_aux = NeuralNetCalssifierComparer(
    input_size, hidden_sizes_classify, hidden_sizes_compare)
model_aux.to(device)
# Loss and optimizer
criterion_main = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_aux.parameters(), lr=lr)


n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (inps, [tgts, clas]) in enumerate(train_loader):
        # origin shape: [n, 2, 14, 14]
        # resized: [n, 2*14*14]
        inps = inps.reshape(-1, 2, 14*14).to(device).float()
        tgts = tgts.to(device).reshape(-1, 1).float()
        cla1 = clas[:, 0].to(device)
        cla2 = clas[:, 1].to(device)
        # Forward pass
        outputs, output_aux_1, output_aux_2 = model_aux(inps)
        loss_main = criterion_main(outputs, tgts)
        loss_axu_1 = criterion_aux(output_aux_1, cla1)
        loss_aux_2 = criterion_aux(output_aux_2, cla2)
        loss_tot = w_main * loss_main + w_aux * loss_axu_1 + w_aux * loss_aux_2

        # Backward and optimize
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        if (i+1) % 100 == 0 and (epoch+1) % 5 == 1:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}]"
                  + f", Loss: {loss_main.item():.4f}")


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (inps, [tgts, clas]) in enumerate(train_loader):
        inps = inps.reshape(-1, 2, 14*14).to(device).float()
        tgts = tgts.to(device).reshape(-1, 1).float()
        predicted = model_aux(inps)[0]
        predicted_cls = predicted.round()
        n_samples += tgts.size(0)
        n_correct += (predicted_cls == tgts).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {data_size} train images: {acc} %')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (inps, [tgts, clas]) in enumerate(test_loader):
        inps = inps.reshape(-1, 2, 14*14).to(device).float()
        tgts = tgts.to(device).reshape(-1, 1).float()
        predicted = model_aux(inps)[0]
        predicted_cls = predicted.round()
        n_samples += tgts.size(0)
        n_correct += (predicted_cls == tgts).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {data_size} test images: {acc} %')
