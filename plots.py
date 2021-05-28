from test import do_mlp_train_test_report, do_cnn_train_test_report
from utils import plot_outputs_single_network_arch_from_list, plot_outputs_single_network_arch_from_list_cnn

if __name__ == '__main__':
    # # calling MLP architectures' train, test, and report
    layers = 3
    layer_widths = [300, 400, 600]
    epochs = 10

    MLP_outputs = {}
    for layer_width in layer_widths:
        hidden_sizes_list = []
        hidden_sizes_list.append([layer_width] * (layers-1) + [196])
        MLP_outputs[layers] = do_mlp_train_test_report(hidden_sizes_list=hidden_sizes_list, epochs=epochs)
    # TODO Ali puts in plotting for his stuff


    epochs = 10
    outputs = {}
    outputs[str(epochs)] = \
        do_cnn_train_test_report(epochs=epochs)
    plot_outputs_single_network_arch_from_list_cnn('CNN_Plots', outputs[str(epochs)], 'CNN Different Architectures', 'CNN', epochs)