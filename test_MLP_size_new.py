import os
import torch
from torch import nn
from runner import *
from net import NeuralNet, NeuralNetCalssifierComparer,\
    NeuralNetCalssifierComparerAuxLoss
from utils import report_from, Verbosity, list_to_string,\
    plot_outputs_single_network_arch_from_list

verbose = Verbosity.No

tensorboard_output = False
if tensorboard_output:
    os.system('rm -rf ./runs >> /dev/null&')
    os.system('tensorboard --logdir=runs --bind_all &')

test_rounds = 1
input_size_cc = 14 * 14  # MLP_RUNNER2
input_size_mlp = 2 * 14 * 14  # MLP_RUNNER
lr = 5e-5
batch_size = 100


def do_train_test(hidden_size_list, epochs, filename):
    MLP_outputs = {}
    MLP_train_losses = {}
    MLP_test_losses = {}
    MLP_train_accs = {}
    MLP_test_accs = {}
    for hidden_size in hidden_sizes_list:
        MLP_run_output = []
        for i in range(test_rounds):
            model_mlp = NeuralNet(input_size_mlp, hidden_size,
                                  batchnorm_bool=True,
                                  dropout_bool=True)  # plain MLP

            optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=lr)
            criterion_mlp = nn.BCELoss()

            MLP = MLPRunner(model_mlp, [criterion_mlp], optimizer_mlp,
                            epochs, batch_size,
                            name=f'MLP_VANILLA_{i}_{hidden_size}',
                            weights=[1.0, 1.0, 1.0],
                            writer_bool=tensorboard_output, verbose=verbose)
            MLP_run_output.append(MLP.run())
            MLP_train_losses[list_to_string(hidden_size)] = MLP.train_loss
            MLP_test_losses[list_to_string(hidden_size)] = MLP.test_loss
            MLP_train_accs[list_to_string(hidden_size)] = MLP.train_acc
            MLP_test_accs[list_to_string(hidden_size)] = MLP.test_acc
            MLP_outputs = {"MLP_train_losses": MLP_train_losses,
                           "MLP_test_losses": MLP_test_losses,
                           "MLP_train_accs": MLP_train_accs,
                           "MLP_test_accs": MLP_test_accs}
        report_from(MLP_run_output, model_mlp, f"MLP_VANILLA_{hidden_size}")
        # plot_outputs_single_network_arch(MLP_outputs,
        #                                  "Vanilla MLP NNs",
        #                                  "MLP 196_")

    CC_outputs = {}
    CC_train_losses = {}
    CC_test_losses = {}
    CC_train_accs = {}
    CC_test_accs = {}
    for hidden_size in hidden_sizes_list:
        CC_run_output = []
        for i in range(test_rounds):
            model_cc = NeuralNetCalssifierComparer(
                input_size_cc,
                hidden_size,
                hidden_sizes_comparer=[80, 80, 20],
                batchnorm_classifer_bool=True,
                dropout_classifier_bool=True)
            optimizer_cc = torch.optim.Adam(model_cc.parameters(), lr=lr)
            criterion_cc = nn.BCELoss()

            CC = MLPClassifierComparerRunner(
                model_cc, [criterion_cc], optimizer_cc,
                epochs, batch_size,
                name=f'MLP_classifier_comparer_{i}_{hidden_size}',
                writer_bool=tensorboard_output, verbose=verbose)
            CC_run_output.append(CC.run())
            CC_train_losses[list_to_string(hidden_size)] = CC.train_loss
            CC_test_losses[list_to_string(hidden_size)] = CC.test_loss
            CC_train_accs[list_to_string(hidden_size)] = CC.train_acc
            CC_test_accs[list_to_string(hidden_size)] = CC.test_acc
            CC_outputs = {"CC_train_losses": CC_train_losses,
                          "CC_test_losses": CC_test_losses,
                          "CC_train_accs": CC_train_accs,
                          "CC_test_accs": CC_test_accs}
        report_from(CC_run_output, model_cc,
                    f"MLP_classifier_comparer_{hidden_size}")
        # plot_outputs_single_network_arch(CC_outputs,
        #                                  "MLP Classifier Comaparer NNs",
        #                                  "MLP CC 196_")

    CC_AUX_outputs = {}
    CC_AUX_train_losses = {}
    CC_AUX_test_losses = {}
    CC_AUX_train_accs = {}
    CC_AUX_test_accs = {}
    for hidden_size in hidden_sizes_list:
        CC_aux_run_output = []
        for i in range(test_rounds):
            model_cc_aux = NeuralNetCalssifierComparerAuxLoss(
                input_size_cc,
                hidden_size,
                hidden_sizes_comparer=[80, 80, 20],
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
                writer_bool=tensorboard_output, verbose=verbose)
            CC_aux_run_output.append(MLP_CC_aux.run())
            CC_AUX_train_losses[list_to_string(
                hidden_size)] = MLP_CC_aux.train_loss
            CC_AUX_test_losses[list_to_string(hidden_size)] =\
                MLP_CC_aux.test_loss
            CC_AUX_train_accs[list_to_string(hidden_size)] =\
                MLP_CC_aux.train_acc
            CC_AUX_test_accs[list_to_string(hidden_size)] = MLP_CC_aux.test_acc
            CC_AUX_outputs = {"CC_AUX_train_losses": CC_AUX_train_losses,
                              "CC_AUX_test_losses": CC_AUX_test_losses,
                              "CC_AUX_train_accs": CC_AUX_train_accs,
                              "CC_AUX_test_accs": CC_AUX_test_accs}
        report_from(CC_aux_run_output,
                    model_cc_aux,
                    f"MLP_classifier_comparer_auxiliary_{hidden_size}")
        # plot_outputs_single_network_arch(
        #     CC_AUX_outputs,
        #     "MLP Classifier Comaparer Auxiliary NNs",
        #     "MLP CC AUX MLP 196_")

    outputs = {}
    outputs.update(MLP_outputs)
    outputs.update(CC_outputs)
    outputs.update(CC_AUX_outputs)
    return outputs


num_epochs = [10, 10]
layers_to_check = [2, 3]
outputs = {}
for layers, epochs in zip(layers_to_check, num_epochs):
    hidden_sizes_list = []
    filename = f"MLP_{layers}_LAYER_ARCHS"
    hidden_sizes_list = []
    layer_widths = [200, 300, 400, 500, 600]
    for lw in layer_widths:
        hidden_sizes_list.append([lw] * (layers-1) + [196])
    outputs[layers] = do_train_test(hidden_sizes_list,
                                    epochs, filename)


def do_plots(outputs, hidden_size_list, epochs, list_archs_for_plot, filename):
    print(f"{list_archs_for_plot=}")
    my_net_len = len(hidden_size_list[0])
    plot_outputs_single_network_arch_from_list(
        filename,
        outputs,
        f"MLP Neural Networks Different Architectures with {my_net_len} Hidden Layers",
        " 196_",
        epochs,
        list_archs_for_plot)


list_list_list_arch_for_plot = [
    [["MLP 196_600_196", "CC 196_400_196", "CC_AUX 196_300_196"],
     ["MLP 196_600_600_196", "CC 196_400_400_196", "CC_AUX 196_300_300_196"],
     ["MLP 196_600_600_600_196",
      "CC 196_400_400_400_196",
      "CC_AUX 196_300_300_300_196"]],
    [["MLP 196_500_196", "CC 196_400_196", "CC_AUX 196_300_196"],
     ["MLP 196_500_500_196", "CC 196_400_400_196", "CC_AUX 196_300_300_196"],
     ["MLP 196_500_500_500_196",
      "CC 196_400_400_400_196",
      "CC_AUX 196_300_300_300_196"]],
    [["MLP 196_500_196", "CC 196_400_196", "CC_AUX 196_300_196"],
     ["MLP 196_500_500_196", "CC 196_400_400_196", "CC_AUX 196_300_300_196"],
     ["MLP 196_500_500_500_196",
      "CC 196_400_400_400_196",
      "CC_AUX 196_300_300_300_196"]]]


for i, list_list_arch_for_plot in enumerate(list_list_list_arch_for_plot):
    for layers, epochs, list_arch_for_plot in zip(layers_to_check,
                                                  num_epochs,
                                                  list_list_arch_for_plot):
        hidden_sizes_list = []
        filename = f"MLP_{layers}_LAYER_ARCHS"
        hidden_sizes_list = []
        layer_widths = [200, 300, 400, 500, 600]
        for lw in layer_widths:
            hidden_sizes_list.append([lw] * (layers-1) + [196])
        if i == 0:
            do_plots(outputs[layers], hidden_sizes_list,
                     epochs, None, filename)
        do_plots(outputs[layers], hidden_sizes_list,
                 epochs, list_arch_for_plot, filename)
