# importing packages and methods from other files in the module
import os
import torch
from torch import nn
from runner import *
from net import *
from utils import report_from, Verbosity, list_to_string,\
    plot_outputs_single_network_arch_from_list_cnn,\
    plot_outputs_single_network_arch_from_list_cnn

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorborad_found = True
except:
    tensorborad_found = False


# Verbosity: Full, Some, No to determine the amount of information
# dumped while training
verbose = Verbosity.No


# tensorboard_output(Boolean) if set to  True it makes Tensorboard output
# in which one  can see the learning curves as well as schematic of the networks
# while the test.py, you will be provided with a link address in the terminal
# simply copy that into your favorite browser and you can visualize the network
# and its training curves
tensorboard_output = True
if tensorboard_output and tensorborad_found:
    # these shell commands get tensorboard up and running
    os.system('rm -rf ./runs >> /dev/null&')
    os.system('tensorboard --logdir=runs --bind_all &')


###########################################################################
########################### HYPER PARAMETERS ##############################
###########################################################################
test_rounds = 10
lr = 1e-4
batch_size = 100
num_epochs = 100
layers = 3  # number of hidden layers for MLPs
layer_widths = [300, 400, 600]  # width of fully connected layers for MLPs
hidden_sizes_comparer = [80, 80, 20]  # the size of hidden layers of compararer
###########################################################################


##########################################################################
#                                                                        #
#                                                                        #
#                                                                        #
#             Train, Test, and report on MLP architectures               #
#                                                                        #
#                                                                        #
#                                                                        #
##########################################################################
def do_mlp_train_test_report(hidden_size_list, epochs, filename):
    input_size_cc = 14 * 14
    input_size_mlp = 2 * 14 * 14
    ###########################################################################
    ####################### PLAIN MLP NETWORK #################################
    ###########################################################################
    MLP_outputs = {}
    for hidden_size in hidden_sizes_list:
        MLP_run_output = []
        for i in range(test_rounds):
            # creating the network
            model_mlp = NeuralNet(input_size_mlp, hidden_size,
                                  batchnorm_bool=True,
                                  dropout_bool=True)  # plain MLP
            # creating the optimizer
            optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=lr)
            # creating loss criterion
            criterion_mlp = nn.BCELoss()
            # creating runner object which does the pre-processing
            # as well as handling train and testing of the network
            MLP = MLPRunner(model_mlp, [criterion_mlp], optimizer_mlp,
                            epochs, batch_size,
                            name=f'MLP_VANILLA_{i}_{hidden_size}',
                            writer_bool=tensorboard_output, verbose=verbose)
            # saving outputs in each test round
            MLP_run_output.append(MLP.run())
            MLP_outputs = {"MLP_train_losses": MLP.train_loss,
                           "MLP_test_losses": MLP.test_loss,
                           "MLP_train_accs": MLP.train_acc,
                           "MLP_test_accs": MLP.test_acc}
        # reporting from the outputs saved
        report_from(MLP_run_output, model_mlp,
                    f"MLP_VANILLA_{hidden_size}")
    ###########################################################################
    ########### MLP NETWORK WITH SEPARATED CNNCalssifierComparer ##############
    ###########################################################################
    CC_outputs = {}
    for hidden_size in hidden_sizes_list:
        CC_run_output = []
        for i in range(test_rounds):
            # creating the network
            model_cc = NeuralNetCalssifierComparer(
                input_size_cc,
                hidden_size,
                hidden_sizes_comparer=hidden_sizes_comparer,
                batchnorm_classifer_bool=True,
                dropout_classifier_bool=True)
            # creating the optimizer
            optimizer_cc = torch.optim.Adam(model_cc.parameters(), lr=lr)
            # creating loss criterion
            criterion_cc = nn.BCELoss()
            # creating runner object which does the pre-processing
            # as well as handling train and testing of the network
            CC = MLPClassifierComparerRunner(
                model_cc, [criterion_cc], optimizer_cc,
                epochs, batch_size,
                name=f'MLP_classifier_comparer_{i}_{hidden_size}',
                writer_bool=tensorboard_output, verbose=verbose)
            CC_run_output.append(CC.run())
            CC_outputs = {"CC_train_losses": CC.train_loss,
                          "CC_test_losses": CC.test_loss,
                          "CC_train_accs": CC.train_acc,
                          "CC_test_accs": CC.test_acc}
        report_from(CC_run_output, model_cc,
                    f"MLP_classifier_comparer_{hidden_size}")

    ###########################################################################
    ########################## CNN NETWORK WITH AUXILIARY LOSS ################
    ###########################################################################
    CC_AUX_outputs = {}
    for hidden_size in hidden_sizes_list:
        CC_aux_run_output = []
        for i in range(test_rounds):
            model_cc_aux = NeuralNetCalssifierComparerAuxLoss(
                input_size_cc,
                hidden_size,
                hidden_sizes_comparer=hidden_sizes_comparer,
                batchnorm_classifer_bool=True,
                dropout_classifier_bool=True)
            optimizer_cc_aux = torch.optim.Adam(
                model_cc_aux.parameters(), lr=lr)
            criterion_cc_aux_main = nn.BCELoss()
            criterion_cc_aux_aux = nn.CrossEntropyLoss()

            MLP_CC_aux = MLPClassifierComparerRunnerAux(
                model_cc_aux, [criterion_cc_aux_main,
                               criterion_cc_aux_aux,
                               criterion_cc_aux_aux], optimizer_cc_aux,
                epochs, batch_size,
                name=f'MLP_classifier_comparer_auxiliary_{i}_{hidden_size}',
                writer_bool=tensorboard_output, verbose=verbose,
                weights=[0.6, 0.2, 0.2])
            CC_aux_run_output.append(MLP_CC_aux.run())
            CC_AUX_outputs = {"CC_AUX_train_losses": MLP_CC_aux.train_loss,
                              "CC_AUX_test_losses": MLP_CC_aux.test_loss,
                              "CC_AUX_train_accs": MLP_CC_aux.train_acc,
                              "CC_AUX_test_accs": MLP_CC_aux.test_acc}
        report_from(CC_aux_run_output, model_cc_aux,
                    f"MLP_classifier_comparer_auxiliary_{hidden_size}")


##########################################################################
#                                                                        #
#                                                                        #
#                                                                        #
#             Train, Test, and report on CNN architectures               #
#                                                                        #
#                                                                        #
#                                                                        #
##########################################################################
def do_cnn_train_test_report(epochs, filename):
    input_size_CC = 14 * 14

    ######################################################################
    ############################ PLAIN CNN NETWORK #######################
    ######################################################################
    CNN_outputs = {}
    CNN_run_output = []
    for i in range(test_rounds):
        # creating the network
        model_CNN = ConvNet_VGG()
        # creating the optimizer
        optimizer_CNN = torch.optim.Adam(model_CNN.parameters(), lr=lr)
        # creating loss criterion
        criterion_CNN = nn.BCELoss()
        # creating runner object which does the pre-processing
        # as well as handling train and testing of the network
        CNN = CNNRunner(model_CNN, [criterion_CNN], optimizer_CNN,
                        epochs, batch_size,
                        name=f'CNN_VANILLA',
                        writer_bool=tensorboard_output, verbose=verbose)
        # saving outputs in each test round
        CNN_run_output.append(CNN.run())
        CNN_outputs = {"CNN_train_losses": CNN.train_loss,
                       "CNN_test_losses": CNN.test_loss,
                       "CNN_train_accs": CNN.train_acc,
                       "CNN_test_accs": CNN.test_acc}
    # reporting from the outputs saved
    report_from(CNN_run_output, model_CNN, f"CNN_VANILLA")

    ###########################################################################
    ########### CNN NETWORK WITH SEPARATED CNNCalssifierComparer ##############
    ###########################################################################
    CNN_CCoutputs = {}
    CNN_CCrun_output = []
    for i in range(test_rounds):
        # creating the network
        model_CNN_CC = CNNCalssifierComparer(
            input_size_CC,
            hidden_sizes_comparer=hidden_sizes_comparer)
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

    ###########################################################################
    ########################## CNN NETWORK WITH AUXILIARY LOSS ################
    ###########################################################################
    CNN_CCAUX_outputs = {}
    CNN_CCAUX_run_output = []
    for i in range(test_rounds):
        # creating the network
        model_CNN_CC_AUX = CNNCalssifierComparerAuxLoss(
            input_size_CC,
            hidden_sizes_comparer=hidden_sizes_comparer)
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
    # calling MLP architectures' train, test, and report
    MLP_outputs = {}
    for layer_width in layer_widths:
        hidden_sizes_list = []
        filename = f"MLP_{layers}_LAYER_ARCHS"
        hidden_sizes_list.append([layer_width] * (layers-1) + [196])
        MLP_outputs[layers] = do_mlp_train_test_report(hidden_sizes_list,
                                                       num_epochs, filename)

    # calling CNN architectures' train, test, and report
    outputs = {}
    filename = f"CNN_ARCHS"
    outputs[str(num_epochs)] = \
        do_cnn_train_test_report(num_epochs, filename)

    for layer_width in layer_widths:
        hidden_sizes_list = []
        filename = f"MLP_{layers}_LAYER_ARCHS"
        hidden_sizes_list.append([layer_width] * (layers-1) + [196])
        MLP_outputs[layers] = do_mlp_train_test_report(hidden_sizes_list,
                                                       num_epochs, filename)
