import abc
from data import load_data
import torch
from utils import Verbosity
from torch.utils.tensorboard import SummaryWriter


class BaseRunner(abc.ABC):
    """
    Base class which controls the training and testing of the various networks.
    """
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 epochs,
                 batch_size,
                 name,
                 weights=[1.0, 1.0, 1.0],
                 writer_bool=False,
                 verbose=Verbosity.No):
        """
        :param model:               Network to train
        :param criterion:           Loss to use
        :param optimizer:           batch SGD optimizer
        :param epochs:              number of epochs
        :param batch_size:          size of batches
        :param name:                name to give to network for summary writer
        :param weights:             if auxiliary loss, weights in order to control loss weights
        :param writer_bool:         whether or not to activate summary writer
        :param verbose:             Full, Some, None to control amount of printed information
        """

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
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
        """
        Run file which calls train, test, print information and saves information.
        :return:                    final train/test loss/accruacy
        """
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
        """
        Calculates the loss. Can be overwritten in case of auxiliary loss.

        :param forward_outputs:     outputs from the forward pass
        :param tgts:                target labels for boolean task
        :param classes:             target labels for multi-class task
        :return:                    loss and output of the final boolean task
        """
        loss = self.criterion[0](forward_outputs, tgts)
        loss.to(self.device)
        return loss, forward_outputs

    def nb_errors(self, output, actual):
        """
        Calculate number of errors for accuracy calculation
        :param output:              Model prediction
        :param actual:              Ground Truth labels
        :return:                    Total number of errors
        """
        if output.size(1) > 1:
            predict = output.argmax(1)
        else:
            predict = output.round()
        error = sum(~predict.eq(actual)).item()
        return error

    def train(self):
        """
        Train model
        :return:                    None
        """
        self.model.train()
        running_loss = 0
        errors = 0
        n_samples = 0
        for i, (inps, [tgts, classes]) in enumerate(self.train_loader):
            # Change size of inputs/labels
            inps, tgts = self.rescale_inputs(inps, tgts)
            inps = inps.to(self.device)
            # Forward pass
            forward_outputs = self.model(inps)

            # Calculate Loss and return correct outputs to calculate error
            loss, outputs = self.apply_criterion(forward_outputs,
                                                 tgts, classes)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Save Info
            running_loss += loss.item()
            errors += self.nb_errors(outputs, tgts)
            n_samples += len(tgts)

        self.train_loss.append(running_loss/len(self.train_loader))
        self.train_acc.append(1 - (errors/n_samples))
        if self.current_epoch % 10 == 0 and self.verbose == Verbosity.Full:
            print(
                f"Epoch: {self.current_epoch}" +
                f"  TRAIN Loss: {self.train_loss[self.current_epoch]:.04f}" +
                f"  TRAIN Accuracy: {self.train_acc[self.current_epoch]:.04f}")
        if self.writer_bool:
            self.writer.add_scalar("Loss/Train",
                                   self.train_loss[self.current_epoch],
                                   self.current_epoch)
            self.writer.add_scalar("Accuracy/Train",
                                   self.train_acc[self.current_epoch],
                                   self.current_epoch)
            self.writer.flush()

    def test(self):
        """
        Test Model
        :return:                    None
        """
        self.model.eval()
        running_loss = 0
        errors = 0
        n_samples = 0
        with torch.no_grad():
            for i, (inps, [tgts, classes]) in enumerate(self.test_loader):

                # Change size of inputs/labels
                inps, tgts = self.rescale_inputs(inps, tgts)
                classes = classes.to(self.device)
                self.model = self.model.to(self.device)
                inps = inps.cuda()

                # Forward and calculate loss
                forward_outputs = self.model(inps)
                loss, outputs = self.apply_criterion(forward_outputs,
                                                     tgts, classes)

                # Save Info
                running_loss += loss.item()
                errors += self.nb_errors(outputs, tgts)
                n_samples += len(tgts)

            self.test_loss.append(running_loss/len(self.test_loader))
            self.test_acc.append(1 - (errors/n_samples))

            if self.current_epoch % 10 == 0 and self.verbose == Verbosity.Full:
                print(
                    f" TEST Loss: {self.test_loss[self.current_epoch]:.04f}" +
                    f" TEST Accuracy: {self.test_acc[self.current_epoch]:.04f}")
            if self.writer_bool:
                self.writer.add_scalar("Loss/Test",
                                       self.test_loss[self.current_epoch],
                                       self.current_epoch)
                self.writer.add_scalar("Accuracy/Test",
                                       self.test_acc[self.current_epoch],
                                       self.current_epoch)
                self.writer.flush()

    def report(self):
        """
        Print final train and test accuracy
        :return:                    None
        """
        print(
            f"{self.name} Neural Network : Ultimate Train Accuracy:{self.train_acc[-1]:.04f}, Test Accuracy:{self.test_acc[-1]:.04f}")

    @abc.abstractmethod
    def rescale_inputs(self, inps, tgts):
        """
        If necessary rescale inputs and target
        :param inps:                 2x14x14 concatenation of the two numbers
        :param tgts:                 boolean labels
        :return:                     None
        """
        pass

    @abc.abstractmethod
    def graph_plot(self):
        """
        Plots the structure of the model in tensorboard
        :return:                    None
        """
        pass

##############################################################################
############ Vanilla Runner ##################################################
##############################################################################


class CNNRunner(BaseRunner):
    """
    Used both for vanilla running and for the classifier comparer running of the CNN's
    """
    def __init__(self, model, criterion, optimizer, epochs,
                 batch_size, name, weights=[1.0],
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
        """
        Reshape targets (batch_size, 1)
        """
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def graph_plot(self):
        pass


class MLPRunner(BaseRunner):
    """
    Vanilla runner for MLP models.
    """
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
        """
        Reshape inputs to be (batch_size, 2*14*14)
        Reshape targets (batch_size, 1)
        """
        inps = inps.reshape(-1, 2 * 14 * 14).to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def graph_plot(self):
        examples = iter(self.train_loader)
        example_data, example_targets = examples.next()

        if self.writer_bool:
            with SummaryWriter(comment='plain mlp') as w:
                self.writer.add_graph(
                    self.model,
                    example_data.reshape(-1, 2 * 14 * 14).to(self.device))

##############################################################################
############ Classifier-Comaprer Runner ######################################
##############################################################################


class MLPClassifierComparerRunner(BaseRunner):
    """
    MLP classifier comparer runner
    """
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
        """
        Reshape inputs to be (batch_size, 2, 14*14)
        Reshape targets (batch_size, 1)
        """
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

##############################################################################
############ Classifier-Comaprer With Auxiliary Loss Runner ##################
##############################################################################


class MLPClassifierComparerRunnerAux(BaseRunner):
    """
    MLP classifier comparerer with auxiliary loss runner
    """
    def __init__(self, model, criterion, optimizer,
                 epochs, batch_size, name,
                 weights=[1.0, 1.0, 1.0],
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
        self.weights = weights

    def apply_criterion(self, forward_outputs, tgts, classes):
        """
        Need to overwrite this to properly apply auxiliary losses
        """
        tgts = tgts.to(self.device)
        classes = classes.to(self.device)

        # Create list of targets that match order of our losses
        tgtss = [tgts,
                 classes[:, 0],
                 classes[:, 1]]
        loss = torch.Tensor([0.0]).to(self.device)

        # Calculate losses for our two auxiliary and main loss. Weight them accordingly
        for i, outputs in enumerate(forward_outputs):
            loss += self.weights[i] * self.criterion[i](outputs, tgtss[i])
        ret_output = forward_outputs[0].to(self.device)

        return loss, ret_output

    def rescale_inputs(self, inps, tgts):
        """
        Reshape inputs to be (batch_size, 2, 14*14)
        Reshape targets (batch_size, 1)
        """
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

class CNNClassifierComparerRunnerAux(BaseRunner):
    """
    CNN classifier comparer with auxiliary loss runner
    """
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
        """
        Need to overwrite this to properly apply auxiliary losses
        """
        tgts = tgts.to(self.device)
        classes = classes.to(self.device)

        # Create list of targets that match order of our losses
        tgtss = [tgts,
                 classes[:, 0],
                 classes[:, 1]]
        loss = torch.Tensor([0.0]).to(self.device)

        # Calculate losses for our two auxiliary and main loss. Weight them accordingly
        for i, outputs in enumerate(forward_outputs):
            loss += self.criterion[i](outputs, tgtss[i])

        return loss, forward_outputs[0]

    def rescale_inputs(self, inps, tgts):
        """
        Reshape targets (batch_size, 1)
        """
        inps = inps.to(self.device).float()
        tgts = tgts.to(self.device).reshape(-1, 1).float()
        return inps, tgts

    def graph_plot(self):
        examples = iter(self.train_loader)
        example_data, example_targets = examples.next()
        if self.writer_bool:
            with SummaryWriter(comment='classifier_comparer') as w:
                self.writer.add_graph(
                    self.model,
                    example_data.to(self.device))
