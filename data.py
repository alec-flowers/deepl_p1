import dlc_practical_prologue as prologue
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    """
    Creates dataset with both boolean and class labels. Used for auxiliary loss purposes
    """
    def __init__(self, inputs, targets, classes):
        """
        :param inputs:          Images 2x14x14
        :param targets:         boolean of greater than or not
        :param classes:         classes of the two images
        """
        super(Dataset, self).__init__()
        self.data = []
        for inp, tgt, cla in zip(inputs, targets, classes):
            self.data.append([inp, [tgt, cla]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


def load_data(batch_size):
    """
    Load data into DataLoaders using the loader provided in class.
    :param batch_size:          Size of batches for data
    :return:                    train dataset, test dataset
    """
    train_data, train_labels, train_classes, test_data, test_labels, test_classes = prologue.generate_pair_sets(1000)

    train_dataset = CustomDataset(train_data, train_labels, train_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomDataset(test_data, test_labels, test_classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
