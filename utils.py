from cycler import cycler
from matplotlib.colors import hsv_to_rgb
import torch
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
from matplotlib import rc
font = {'size': 16}
matplotlib.rc('font', **font)

colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
          for i in range(1000)]
plt.rc('axes', prop_cycle=(cycler('color', colors)))


class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


class Verbosity(Enum):
    Full = 2
    Some = 1
    No = 0


def report_from(run_output, name):
    rounds = len(run_output)
    train_loss = torch.zeros(rounds)
    test_loss = torch.zeros(rounds)
    train_accs = torch.zeros(rounds)
    test_accs = torch.zeros(rounds)
    for i in range(rounds):
        train_loss[i], test_loss[i],\
            train_accs[i], test_accs[i] =\
            (run_output[i])
    std_test_acc, mean_test_acc = torch.std_mean(test_accs)
    std_train_acc, mean_train_acc = torch.std_mean(train_accs)
    if rounds > 1:
        print(
            f"TRAIN ACCURACIES for {name} MEAN: {mean_train_acc.item():.04f}, STD: {std_train_acc.item():.04f}")
        print(
            f"TEST ACCURACIES for {name} MEAN: {mean_test_acc.item():.04f}, STD: {std_test_acc.item():.04f}")
    else:
        print(f"TRAIN ACCURACY for {name}: {mean_train_acc.item():.04f}")
        print(f"TEST ACCURACY for {name}: {mean_test_acc.item():.04f}")
    print("\n")


def list_to_string(input_list):
    # initialize an empty string
    return_string = '_'.join([str(elem) for elem in input_list])
    # return string
    return return_string


def plot_outputs_single_network_arch_from_list(
    filename_inp, outputs, title, label,
    epochs, list_labels=None):
    fig1 = plt.figure(figsize=[24, 12])
    titles = ["train loss", "test loss",
              "train accuracy", "test accuracy"]
    linestyles = ["-", ":", "-.", "--"]

    for j, output_key in enumerate(outputs):
        output_val = outputs[output_key]
        out_key = output_key.replace('_train_accs', "")
        out_key = out_key.replace('_test_accs', "")
        out_key = out_key.replace('_test_losses', "")
        out_key = out_key.replace('_train_losses', "")
        counter = j % 4
        NN_type = int(j / 4)
        ax = plt.subplot(2, 2, counter+1)
        handles, labels = [], []
        for i, key in enumerate(output_val):
            this_label = out_key + label + key
            plot_bool = False
            if list_labels == None:
                plot_bool = True
            else:
                if this_label in list_labels:
                    plot_bool = True

            if plot_bool:
                ax.plot(output_val[key],
                        label=this_label,
                        ls=linestyles[NN_type])
                ax.set_title(titles[counter])
                handles, labels = ax.get_legend_handles_labels()
                ax.grid("Major")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlim([0, epochs])
                if counter > 1:
                    ax.set_ylim([0.3, 1.0])
    fig1.suptitle(title, fontsize=22)
    fig1.legend(handles, labels, bbox_to_anchor=(
        1.1, 0.5), loc='center right', fontsize=18)
    if list_labels == None:
        filename = filename_inp + ".svg"
    else:
        filename = filename_inp + list_to_string(list_labels) + ".svg"
    fig1.savefig(filename, format="svg", bbox_inches="tight")
    # plt.show()
