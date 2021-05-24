import abc
from data import load_data
import torch
from utils import nb_errors, DummySummaryWriter, Verbosity
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
                 writer_bool=False,
                 verbose=Verbosity.No):

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
        self.writer_bool = writer_bool
        self.verbose = verbose
        if writer_bool:
            self.writer = SummaryWriter('runs/'+self.name)

        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

    def run(self):
        for e in range(self.epochs):
            self.train()
            self.test()
            self.current_epoch += 1
        if self.verbose == Verbosity.Some or self.verbose == Verbosity.Full:
            self.report()
        if self.writer_bool:
            self.graph_plot()
        return (self.train_loss[-1], self.test_loss[-1],
                self.train_acc[-1], self.test_acc[-1])

    def apply_criterion(self, forward_outputs, tgts, classes):
        loss = self.criterion[0](forward_outputs, tgts)
        loss.to(self.device)
        return loss, forward_outputs

    def train(self):
        self.model.train()
        running_loss = 0
        errors = 0
        n_samples = 0
        for i, (inps, [tgts, classes]) in enumerate(self.train_loader):
            # origin shape: [n, 2, 14, 14]
            # resized: [n, 2*14*14]
            classes = classes.to(self.device)
            inps, tgts_rescaled = self.rescale_inputs(inps, tgts)
            tgts_rescaled.to(self.device)
            tgts = tgts.to(self.device)
            # Forward pass
            forward_outputs = self.model(inps)
            loss, outputs = self.apply_criterion(forward_outputs,
                                                 tgts_rescaled, classes)
            outputs.to(self.device)
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
        if self.current_epoch % 10 == 0 and self.verbose == Verbosity.Full:
            print(
                f"Epoch: {self.current_epoch}" +
                f"  Loss: {self.train_loss[self.current_epoch]:.04f}" +
                f"  Accuracy: {self.train_acc[self.current_epoch]:.04f}")
        if self.writer_bool:
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
            for i, (inps, [tgts, classes]) in enumerate(self.test_loader):
                inps, tgts = self.rescale_inputs(inps, tgts)
                classes = classes.to(self.device)
                tgtss = [tgts,
                         classes[:, 0],
                         classes[:, 1]]
                forward_outputs = self.model(inps)
                loss, outputs = self.apply_criterion(forward_outputs,
                                                     tgts, classes)

                # Save Info
                running_loss += loss.item()
                errors += nb_errors(outputs, tgts)
                n_samples += len(tgts)

            self.test_loss.append(running_loss/len(self.test_loader))
            self.test_acc.append(1 - (errors/n_samples))

            if self.current_epoch % 10 == 0 and self.verbose == 2:
                print(
                    f" TRAIN_Loss: {self.test_loss[self.current_epoch]:.04f}" +
                    f" Accuracy: {self.test_acc[self.current_epoch]:.04f}")
            if self.writer_bool:
                self.writer.add_scalar("Loss/Test",
                                       self.test_loss[self.current_epoch],
                                       self.current_epoch)
                self.writer.add_scalar("Accuracy/Test",
                                       self.test_acc[self.current_epoch],
                                       self.current_epoch)
                self.writer.flush()

    def report(self):
        print(
            f"{self.name} Neural Network : Ultimate Train Accuracy:{self.train_acc[-1]:.04f}, Test Accuracy:{self.test_acc[-1]:.04f}")

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
        return inps, tgts

    def graph_plot(self):
        pass


class MLPRunner(BaseRunner):
    def __init__(self, model, criterion, optimizer,
                 epochs, batch_size, name, weights=[1.0],
                 writer_bool=False, verbose=Verbosity.No):
        super().__init__(model,
                         criterion,
                         optimizer,
                         epochs,
                         batch_size,
                         name,
                         weights,
                         writer_bool,
                         verbose)

    def rescale_inputs(self, inps, tgts):
        inps = inps.reshape(-1, 2 * 14 * 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def reshape_inputs(self, inps):
        return inps.reshape(-1, 2, 14, 14)

    def graph_plot(self):
        examples = iter(self.train_loader)
        example_data, example_targets = examples.next()

        if self.writer_bool:
            with SummaryWriter(comment='plain mlp') as w:
                self.writer.add_graph(
                    self.model,
                    example_data.reshape(-1, 2 * 14 * 14).to(self.device))


class MLPClassifierComparerRunner(BaseRunner):
    def __init__(self, model, criterion, optimizer,
                 epochs, batch_size, name, weights=[1.0],
                 writer_bool=False, verbose=Verbosity.No):
        super().__init__(model,
                         criterion,
                         optimizer,
                         epochs,
                         batch_size,
                         name,
                         weights,
                         writer_bool,
                         verbose)

    def rescale_inputs(self, inps, tgts):
        inps = inps.reshape(-1, 2, 14 * 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def graph_plot(self):
        examples = iter(self.train_loader)
        example_data, example_targets = examples.next()
        if self.writer_bool:
            with SummaryWriter(comment='classifier_comparer') as w:
                self.writer.add_graph(
                    self.model,
                    example_data.reshape(-1, 2, 14 * 14).to(self.device))


class MLPClassifierComparerRunnerAux(BaseRunner):
    def __init__(self, model, criterion, optimizer,
                 epochs, batch_size, name, weights=[1.0],
                 writer_bool=False, verbose=Verbosity.No):
        super().__init__(model,
                         criterion,
                         optimizer,
                         epochs,
                         batch_size,
                         name,
                         weights,
                         writer_bool,
                         verbose)

    def apply_criterion(self, forward_outputs, tgts, classes):
        tgtss = [tgts,
                 classes[:, 0],
                 classes[:, 1]]
        loss = torch.Tensor([0.0]).to(self.device)
        for i, outputs in enumerate(forward_outputs):
            loss += self.criterion[i](outputs, tgtss[i])

        return loss, forward_outputs[0]

    def rescale_inputs(self, inps, tgts):
        inps = inps.reshape(-1, 2, 14 * 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def graph_plot(self):
        examples = iter(self.train_loader)
        example_data, example_targets = examples.next()
        if self.writer_bool:
            with SummaryWriter(comment='classifier_comparer') as w:
                self.writer.add_graph(
                    self.model,
                    example_data.reshape(-1, 2, 14 * 14).to(self.device))
