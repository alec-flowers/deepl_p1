from test import do_mlp_train_test_report, do_cnn_train_test_report
from utils import plot_outputs_single_network_arch_from_list,\
    plot_outputs_single_network_arch_from_list_cnn


def do_plots(outputs, hidden_size_list, epochs, list_archs_for_plot, filename):
    print(f"{list_archs_for_plot=}")
    my_net_len = len(hidden_size_list[0])
    plot_outputs_single_network_arch_from_list(
        filename,
        outputs,
        f"MLP Neural Networks Different Architectures with {my_net_len} Hidden Layers",
        " 196_", epochs,
        list_archs_for_plot)


if __name__ == '__main__':
    # calling MLP architectures' train, test, and report
    layers = 3
    layer_widths = [300, 400, 600]
    epochs = 100

    MLP_outputs = {}
    for layer_width in layer_widths:
        hidden_sizes_list = []
        hidden_sizes_list.append([layer_width] * (layers-1) + [196])
        MLP_outputs[layers] = do_mlp_train_test_report(
            hidden_sizes_list=hidden_sizes_list, epochs=epochs)
    plot_list = [
        [["MLP 196_600_600_196", "CC 196_400_400_196", "CC_AUX 196_300_300_196"]],
        [["MLP 196_600_600_196", "CC 196_600_600_196", "CC_AUX 196_600_600_196"]]]

    hidden_sizes_list = []
    for i, list_list_arch_for_plot in enumerate(plot_list):
        for list_arch_for_plot in zip(list_list_arch_for_plot):
            filename = f"MLP_{layers}_LAYER_ARCHS"
            hidden_sizes_list.append([layer_width] * (layers-1) + [196])
            for lw in layer_widths:
                hidden_sizes_list.append([lw] * (layers-1) + [196])
        do_plots(MLP_outputs[layers], hidden_sizes_list,
                 epochs, list_arch_for_plot, filename)

    epochs = 100
    outputs = {}
    outputs[str(epochs)] = \
        do_cnn_train_test_report(epochs=epochs)
    plot_outputs_single_network_arch_from_list_cnn(
        'CNN_Plots',
        outputs[str(epochs)],
        'CNN Different Architectures',
        'CNN', epochs)
