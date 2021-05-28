import os
import torch
from torch import nn
from runner import *
from net import *
from utils import report_from, Verbosity, list_to_string,\
    plot_outputs_single_network_arch_from_list_cnn
import pickle

# Verbosity: Full, Some, No to determine the amount of information dumped while training
verbose = Verbosity.No

# tensorboard_output is Boolean if True it makes Tensorboard output in which one can see the learning curves as well as schematic of the networks
# while the test.py, you will be provided with a http address in the terminal simply copy that into your favorite browser and you can visualize the network and its training curves
tensorboard_output = False
if tensorboard_output:
    # these shell commands get tensorboard up and running
    os.system('rm -rf ./runs >> /dev/null&')
    os.system('tensorboard --logdir=runs --bind_all &')

# Hyper-Paramters
test_rounds = 10
lr = 1e-4
batch_size = 100
num_epochs = 100


def do_train_test(epochs, filename):
    ########### PLAIN CNN NETWORK ##############
    CNN_outputs = {}
    CNN_run_output = []
    for i in range(test_rounds):
        # creating the network
        model_CNN = ConvNet4()
        # creating the optimizer
        optimizer_CNN = torch.optim.Adam(model_CNN.parameters(), lr=lr)
        # creating loss criterion
        criterion_CNN = nn.BCELoss()
        # creating runner object which does the pre-processing
        # as well as handling train and testing of the network
        CNN = CNNRunner(model_CNN, [criterion_CNN], optimizer_CNN,
                        epochs, batch_size,
                        name=f'CNN_VANILLA',
                        weights=[1.0, 1.0, 1.0],
                        writer_bool=tensorboard_output, verbose=verbose)
        # saving outputs in each test round
        CNN_run_output.append(CNN.run())
        CNN_outputs = {"CNN_train_losses": CNN.train_loss,
                       "CNN_test_losses": CNN.test_loss,
                       "CNN_train_accs": CNN.train_acc,
                       "CNN_test_accs": CNN.test_acc}
    # reporting from the outputs saved
    report_from(CNN_run_output, model_CNN, f"CNN_VANILLA")

    ########### CNN NETWORK WITH SEPARATED CNNCalssifierComparer ##############
    CNN_CCoutputs = {}
    CNN_CCrun_output = []
    for i in range(test_rounds):
        input_size_CC = 14 * 14
        # creating the network
        model_CNN_CC = CNNCalssifierComparer(input_size_CC,
                                             hidden_sizes_comparer=[80, 80, 20])
        # creating the optimizer
        optimizer_CC = torch.optim.Adam(model_CNN_CC.parameters(), lr=lr)
        # creating loss criterion
        criterion_CC = nn.BCELoss()
        # creating runner object which does the pre-processing
        # as well as handling train and testing of the network
        CC = CNNRunner(
            model_CNN_CC, [criterion_CC], optimizer_CC,
            epochs, batch_size,
            name=f'CNN_classifier_comparer',
            writer_bool=tensorboard_output, verbose=verbose)
        # saving outputs in each test round
        CNN_CCrun_output.append(CC.run())
        CNN_CCoutputs = {"CNN_CNN_CCtrain_losses": CC.train_loss,
                         "CNN_CNN_CCtest_losses": CC.test_loss,
                         "CNN_CNN_CCtrain_accs": CC.train_acc,
                         "CNN_CNN_CCtest_accs": CC.test_acc}
    # reporting from the outputs saved
    report_from(CNN_CCrun_output, model_CNN_CC, f"CNN_classifier_comparer")

    ########################## CNN NETWORK WITH AUXILIARY LOSS ###############
    CNN_CCAUX_outputs = {}
    CNN_CCAUX_run_output = []
    for i in range(test_rounds):
        input_size_CC = 14 * 14
        # creating the network
        model_CNN_CC_AUX = CNNCalssifierComparerAuxLoss(
            input_size_CC,
            hidden_sizes_comparer=[80, 80, 20])
        # creating the optimizer
        optimizer_CC_AUX = torch.optim.Adam(
            model_CNN_CC_AUX.parameters(), lr=lr)
        # creating loss criteria
        criterion_CC_AUX_main = nn.BCELoss()
        criterion_CC_AUX_AUX = nn.CrossEntropyLoss()
        # creating runner object which does the pre-processing
        # as well as handling train and testing of the network
        CNN_CCAUX = CNNClassifierComparerRunnerAux(
            model_CNN_CC_AUX, [criterion_CC_AUX_main,
                               criterion_CC_AUX_AUX,
                               criterion_CC_AUX_AUX], optimizer_CC_AUX,
            epochs, batch_size,
            name=f'CNN_classifier_comparer_AUXiliary',
            writer_bool=tensorboard_output, verbose=verbose,
            weights=[0.6, 0.2, 0.2])
        # saving outputs in each test round
        CNN_CCAUX_run_output.append(CNN_CCAUX.run())
        CNN_CCAUX_outputs = {"CNN_CCAUX_train_losses": CNN_CCAUX.train_loss,
                             "CNN_CCAUX_test_losses": CNN_CCAUX.test_loss,
                             "CNN_CCAUX_train_accs": CNN_CCAUX.train_acc,
                             "CNN_CCAUX_test_accs": CNN_CCAUX.test_acc}
    # reporting from the outputs saved
    report_from(CNN_CCAUX_run_output, model_CNN_CC_AUX,
                f"CNN_classifier_comparer_auxiliary")

    outputs = {}
    outputs.update(CNN_outputs)
    outputs.update(CNN_CCoutputs)
    outputs.update(CNN_CCAUX_outputs)
    return outputs


if __name__ == '__main__':
    outputs = {}
    filename = f"CNN_plots"
    outputs[str(num_epochs)] = do_train_test(num_epochs, filename)
    plot_outputs_single_network_arch_from_list_cnn(
        'CNN_Plots', outputs['100'],
        'CNN Different Architectures', 'CNN', num_epochs)
