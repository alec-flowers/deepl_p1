import os
import torch
from torch import nn
from runner import *
from net import *
from utils import report_from, Verbosity, list_to_string,\
    plot_outputs_single_network_arch_from_list
import pickle

verbose = Verbosity.No

tensorboard_output = False
if tensorboard_output:
    os.system('rm -rf ./runs >> /dev/null&')
    os.system('tensorboard --logdir=runs --bind_all &')

test_rounds = 10
lr = 1e-4
batch_size = 100


def do_train_test(epochs, filename):
    CNN_outputs = {}
    CNN_train_losses = {}
    CNN_test_losses = {}
    CNN_train_accs = {}
    CNN_test_accs = {}
    CNN_run_output = []
    for i in range(test_rounds):
        model_CNN = ConvNet4()

        optimizer_CNN = torch.optim.Adam(model_CNN.parameters(), lr=lr)
        criterion_CNN = nn.BCELoss()

        CNN = ConvRunner(model_CNN, [criterion_CNN], optimizer_CNN,
                        epochs, batch_size,
                        name=f'MLP_VANILLA',
                        weights=[1.0, 1.0, 1.0],
                        writer_bool=tensorboard_output, verbose=verbose)
        CNN_run_output.append(CNN.run())
        CNN_train_losses['CNN'] = CNN.train_loss
        CNN_test_losses['CNN'] = CNN.test_loss
        CNN_train_accs['CNN'] = CNN.train_acc
        CNN_test_accs['CNN'] = CNN.test_acc
        CNN_outputs = {"CNN_train_losses": CNN_train_losses,
                       "CNN_test_losses": CNN_test_losses,
                       "CNN_train_accs": CNN_train_accs,
                       "CNN_test_accs": CNN_test_accs}
    report_from(CNN_run_output, f"CNN_VANILLA")

    CC_outputs = {}
    CC_train_losses = {}
    CC_test_losses = {}
    CC_train_accs = {}
    CC_test_accs = {}
    CC_run_output = []
    for i in range(test_rounds):
        input_size_cc = 14 * 14
        model_cc = CNNCalssifierComparer(input_size_cc,
                                             hidden_sizes_comparer=[80, 80, 20])
        optimizer_cc = torch.optim.Adam(model_cc.parameters(), lr=lr)
        criterion_cc = nn.BCELoss()

        CC = ConvRunner(
            model_cc, [criterion_cc], optimizer_cc,
            epochs, batch_size,
            name=f'CNN_classifier_comparer',
            writer_bool=tensorboard_output, verbose=verbose)
        CC_run_output.append(CC.run())
        CC_train_losses['CNN_CC'] = CC.train_loss
        CC_test_losses['CNN_CC'] = CC.test_loss
        CC_train_accs['CNN_CC'] = CC.train_acc
        CC_test_accs['CNN_CC'] = CC.test_acc
        CC_outputs = {"CNN_CC_train_losses": CC_train_losses,
                      "CNN_CC_test_losses": CC_test_losses,
                      "CNN_CC_train_accs": CC_train_accs,
                      "CNN_CC_test_accs": CC_test_accs}
    report_from(CC_run_output, f"CNN_classifier_comparer")


    CC_AUX_outputs = {}
    CC_AUX_train_losses = {}
    CC_AUX_test_losses = {}
    CC_AUX_train_accs = {}
    CC_AUX_test_accs = {}
    CC_aux_run_output = []
    for i in range(test_rounds):
        model_cc_aux = CNNCalssifierComparerAuxLoss(
            input_size_cc,
            hidden_sizes_comparer=[80, 80, 20])
        optimizer_cc_aux = torch.optim.Adam(
            model_cc_aux.parameters(), lr=lr)
        criterion_cc_aux_main = nn.BCELoss()
        criterion_cc_aux_aux = nn.CrossEntropyLoss()

        MLP_CC_aux = CNNClassifierComparerRunnerAux(
            model_cc_aux, [criterion_cc_aux_main,
                           criterion_cc_aux_aux,
                           criterion_cc_aux_aux], optimizer_cc_aux,
            epochs, batch_size,
            name=f'CNN_classifier_comparer_auxiliary',
            writer_bool=tensorboard_output, verbose=verbose,
            weights = [0.6, 0.2, 0.2])
        CC_aux_run_output.append(MLP_CC_aux.run())
        CC_AUX_train_losses['CNN_CC_AUX'] = MLP_CC_aux.train_loss
        CC_AUX_test_losses['CNN_CC_AUX'] =\
            MLP_CC_aux.test_loss
        CC_AUX_train_accs['CNN_CC_AUX'] =\
            MLP_CC_aux.train_acc
        CC_AUX_test_accs['CNN_CC_AUX'] = MLP_CC_aux.test_acc
        CC_AUX_outputs = {"CC_AUX_train_losses": CC_AUX_train_losses,
                          "CC_AUX_test_losses": CC_AUX_test_losses,
                          "CC_AUX_train_accs": CC_AUX_train_accs,
                          "CC_AUX_test_accs": CC_AUX_test_accs}
    report_from(CC_aux_run_output,
                f"CNN_classifier_comparer_auxiliary")

    outputs = {}
    outputs.update(CNN_outputs)
    outputs.update(CC_outputs)
    outputs.update(CC_AUX_outputs)
    return outputs


if __name__ == '__main__':
    num_epochs = [100]
    outputs = {}
    for epochs in num_epochs:
        filename = f"CNN_plots"
        outputs[str(epochs)] = do_train_test(epochs, filename)
    # a_file = open("cnn_data_01.pkl", "rb")
    # outputs = pickle.load(a_file)
    # a_file.close()
    #
    # plot_outputs_single_network_arch_from_list('CNN_Plots', outputs['200'], 'CNN Different Architectures', 'CNN', 100)