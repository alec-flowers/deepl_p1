import abc
from data import load_data
import torch
from utils import nb_errors
from torch import nn


class BaseRunner(abc.ABC):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 summarywriter,
                 epochs,
                 batch_size):

        self.model = model
        self.batch_size = batch_size
        self.trainloader, self.testloader = load_data(self.batch_size)
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = summarywriter
        self.epochs = epochs
        self.current_epoch = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

    def run(self):
        for e in range(self.epochs):
            self.train()
            self.test()
            self.current_epoch += 1

    def train(self):
        self.model.train()
        running_loss = 0
        errors = 0
        n_samples = 0
        for i, (inps, [tgts, clas]) in enumerate(self.trainloader):
            # origin shape: [n, 2, 14, 14]
            # resized: [n, 2*14*14]
            inps, tgts = self.rescale_inputs(inps, tgts)

            # Forward pass
            outputs = self.model(inps)
            loss = self.criterion(outputs, tgts)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #Save Info
            running_loss += loss.item()
            errors += nb_errors(outputs, tgts)
            n_samples += len(tgts)

        a = len(self.trainloader)
        self.train_loss.append(running_loss/len(self.trainloader))
        self.train_acc.append(1 - (errors/n_samples))
        #self.writer.add_scalar("Loss/Train", running_loss / batch_size, e)
        if self.current_epoch % 10 == 0:
            print(f'Epoch: {self.current_epoch}  Loss: {self.train_loss[self.current_epoch]:.04f}  Accuracy: {self.train_acc[self.current_epoch]:.04f}')

        #self.writer.flush()

    def test(self):
        self.model.eval()
        running_loss = 0
        errors = 0
        n_samples = 0
        with torch.no_grad():
            for i, (inps, [tgts, clas]) in enumerate(self.testloader):
                inps, tgts = self.rescale_inputs(inps, tgts)
                outputs = self.model(inps)
                loss = self.criterion(outputs, tgts)

                #Save Info
                running_loss += loss.item()
                errors += nb_errors(outputs, tgts)
                n_samples += len(tgts)

            self.test_loss.append(running_loss/len(self.testloader))
            self.test_acc.append(1 - (errors/n_samples))
            #self.writer.add_scalar("Loss/Train", running_loss / batch_size, e)
            if self.current_epoch % 10 == 0:
                print(f'    TRAIN Loss: {self.test_loss[self.current_epoch]:.04f}  Accuracy: {self.test_acc[self.current_epoch]:.04f}')

    @abc.abstractmethod
    def rescale_inputs(self, inps, tgts):
        pass


class ConvRunner(BaseRunner):
    def __init__(self, model, criterion, optimizer, summarywriter, epochs, batch_size):
        super().__init__(model,
                         criterion,
                         optimizer,
                         summarywriter,
                         epochs,
                         batch_size)

    def rescale_inputs(self, inps, tgts):
        return inps, tgts


class MLPRunner(BaseRunner):
    def __init__(self, model, criterion, optimizer, summarywriter, epochs, batch_size):
        super().__init__(model,
                         criterion,
                         optimizer,
                         summarywriter,
                         epochs,
                         batch_size)

    def rescale_inputs(self, inps, tgts):
        inps = inps.reshape(-1, 2 * 14 * 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts


class MLP2Runner(BaseRunner):
    def __init__(self, model, criterion, optimizer, summarywriter, epochs, batch_size):
        super().__init__(model,
                         criterion,
                         optimizer,
                         summarywriter,
                         epochs,
                         batch_size)

    def rescale_inputs(self, inps, tgts):
        inps = inps.reshape(-1, 2, 14 * 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts