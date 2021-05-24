import abc
from data import load_data
import torch
from utils import nb_errors, DummySummaryWriter
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class BaseRunner(abc.ABC):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 epochs,
                 batch_size,
                 name,
                 weights=[1.0],
                 writer_bool=False):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.name = name
        self.model = model
        self.model.to(self.device)
        self.batch_size = batch_size
        self.train_loader, self.test_loader = load_data(self.batch_size)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.current_epoch = 0
        self.weights = weights
        if writer_bool:
            self.writer = SummaryWriter('runs/'+self.name)
        else:
            self.writer = DummySummaryWriter()

        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

    def run(self):
        for e in range(self.epochs):
            self.train()
            self.test()
            self.current_epoch += 1
        self.graph_plot()
        print("DONE")

    def train(self):
        self.model.train()
        running_loss = 0
        errors = 0
        n_samples = 0
        for i, (inps, [tgts, clas]) in enumerate(self.train_loader):
            # origin shape: [n, 2, 14, 14]
            # resized: [n, 2*14*14]
            clas = clas.to(self.device)
            inps, tgts = self.rescale_inputs(inps, tgts)
            tgtss = [tgts,
                     clas[:, 0],
                     clas[:, 1]]
            # Forward pass
            outputss = self.model(inps)
            self.len_outputss = \
                1 if type(outputss) is not tuple else len(outputss)
            if self.len_outputss == 1:
                loss = self.criterion[0](outputss, tgts)
                outputs = outputss
            else:
                loss = torch.Tensor([0.0]).to(self.device)
                for i, outputs in enumerate(outputss):
                    # print(f"{i=}")
                    # print(f"{tgtss[i].shape=}")
                    # print(f"{outputs.shape=}")
                    # print(f"{self.criterion[i]=}")
                    loss += self.criterion[i](outputs, tgtss[i])
                outputs = outputss[0]

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Save Info
            running_loss += loss.item()
            errors += nb_errors(outputs, tgts)
            n_samples += len(tgts)

        a = len(self.train_loader)
        self.train_loss.append(running_loss/len(self.train_loader))
        self.train_acc.append(1 - (errors/n_samples))
        if self.current_epoch % 10 == 0:
            print(
                f"Epoch: {self.current_epoch}" +
                f"  Loss: {self.train_loss[self.current_epoch]:.04f}" +
                f"  Accuracy: {self.train_acc[self.current_epoch]:.04f}")

        self.writer.add_scalar("Loss/Train",
                               self.train_loss[self.current_epoch],
                               self.current_epoch)
        self.writer.add_scalar("Accuracy/Train",
                               self.train_acc[self.current_epoch],
                               self.current_epoch)
        self.writer.flush()

    def test(self):
        self.model.eval()
        running_loss = 0
        errors = 0
        n_samples = 0
        with torch.no_grad():
            for i, (inps, [tgts, clas]) in enumerate(self.test_loader):
                inps, tgts = self.rescale_inputs(inps, tgts)
                clas = clas.to(self.device)
                tgtss = [tgts,
                         clas[:, 0],
                         clas[:, 1]]
                outputss = self.model(inps)
                if self.len_outputss == 1:
                    loss = self.criterion[0](outputss, tgts)
                    outputs = outputss
                else:
                    loss = torch.Tensor([0.0]).to(self.device)
                    for i, outputs in enumerate(outputss):
                        loss += self.criterion[i](outputs, tgtss[i])
                    outputs = outputss[0]

                # Save Info
                running_loss += loss.item()
                errors += nb_errors(outputs, tgts)
                n_samples += len(tgts)

            self.test_loss.append(running_loss/len(self.test_loader))
            self.test_acc.append(1 - (errors/n_samples))

            if self.current_epoch % 10 == 0:
                print(
                    f" TRAIN_Loss: {self.test_loss[self.current_epoch]:.04f}" +
                    f"  Accuracy: {self.test_acc[self.current_epoch]:.04f}")

            self.writer.add_scalar("Loss/Test",
                                   self.test_loss[self.current_epoch],
                                   self.current_epoch)
            self.writer.add_scalar("Accuracy/Test",
                                   self.test_acc[self.current_epoch],
                                   self.current_epoch)
            self.writer.flush()

    @abc.abstractmethod
    def rescale_inputs(self, inps, tgts):
        pass

    @abc.abstractmethod
    def graph_plot(self):
        pass


class ConvRunner(BaseRunner):
    def __init__(self, model, criterion, optimizer, epochs,
                 batch_size, name, weights=[1.0], writer_bool=False):
        super().__init__(model,
                         criterion,
                         optimizer,
                         epochs,
                         batch_size,
                         name,
                         weights,
                         writer_bool)

    def rescale_inputs(self, inps, tgts):
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def graph_plot(self):
        pass


class MLPRunner(BaseRunner):
    def __init__(self, model, criterion, optimizer,
                 epochs, batch_size, name, weights=[1.0], writer_bool=False):
        super().__init__(model,
                         criterion,
                         optimizer,
                         epochs,
                         batch_size,
                         name,
                         weights,
                         writer_bool)

    def rescale_inputs(self, inps, tgts):
        # TODO changed this with an extra 1
        inps = inps.reshape(-1, 1, 2 * 14 * 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def reshape_inputs(self, inps):
        return inps.reshape(-1, 2, 14, 14)

    def graph_plot(self):
        examples = iter(self.train_loader)
        example_data, example_targets = examples.next()

        with SummaryWriter(comment='plain mlp') as w:
            self.writer.add_graph(
                self.model,
                example_data.reshape(-1, 2 * 14 * 14).to(self.device))


class MLP2Runner(BaseRunner):
    def __init__(self, model, criterion, optimizer,
                 epochs, batch_size, name, weights=[1.0], writer_bool=False):
        super().__init__(model,
                         criterion,
                         optimizer,
                         epochs,
                         batch_size,
                         name,
                         weights,
                         writer_bool)

    def rescale_inputs(self, inps, tgts):
        inps = inps.reshape(-1, 2, 14 * 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

class CNNRunner(BaseRunner):
    def __init__(self, model, criterion, optimizer,
                 epochs, batch_size, name, weights=[1.0], writer_bool=False):
        super().__init__(model,
                         criterion,
                         optimizer,
                         epochs,
                         batch_size,
                         name,
                         weights,
                         writer_bool)

    def rescale_inputs(self, inps, tgts):
        #inps = inps.reshape(-1, 2, 14, 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def graph_plot(self):
        examples = iter(self.train_loader)
        example_data, example_targets = examples.next()
        with SummaryWriter(comment='classifier_comparer') as w:
            self.writer.add_graph(
                self.model,
                example_data.reshape(-1, 2, 14 * 14).to(self.device)
            )
