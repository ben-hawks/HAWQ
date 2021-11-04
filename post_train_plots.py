from torch import nn
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import os
import os.path as path

def plot_kernels(model, text="", output=None):
    weight_plt = plt.figure()
    weight_ax = weight_plt.add_subplot()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, qnn.QuantLinear):
            weights = module.weight
            weights = weights.reshape(-1).detach().cpu().numpy()
            weight_ax.hist(abs(weights), bins=10 ** np.linspace(np.log10(0.0001), np.log10(2.5), 100), alpha=0.6, label=name)
    weight_ax.legend(loc='upper left')
    weight_ax.set_xscale('log')
    if model.quantized_model:
        precision = model.weight_precision
        weight_ax.set_title("Quant (" + str(precision) + "b) Model Weights " + text)
    else:
        weight_ax.set_title("Float Model Weights " + text)
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    weight_ax.set_ylabel("Number of Weights")
    weight_ax.set_xlabel("Absolute Weights")
    if output is None:
        os.makedirs('weight_dists/',exist_ok=True)
        output = os.path.join('weight_dists/', ('weight_dist_' + str(time) + '.png'))
    weight_plt.savefig(output)
    weight_plt.show()
    plt.close(weight_plt)

def plot_total_loss(model_set, model_totalloss_set, model_estop_set, outputDir='..'):
    # Total loss over fine tuning
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    for model, model_loss, model_estop in zip(model_set, model_totalloss_set, model_estop_set):
        tloss_plt = plt.figure()
        tloss_ax = tloss_plt.add_subplot()
        filename = 'total_loss_{}.png'.format(time)
        tloss_ax.plot(range(1, len(model_loss) + 1), model_loss, label='Training Loss')
        #tloss_ax.plot(range(1, len(model_loss[1]) + 1), model_loss[1], label='Validation Loss')
        # plot each stopping point
        for stop in model_estop:
            tloss_ax.axvline(stop+1, linestyle='--', color='r', alpha=0.3)
        tloss_ax.set_xlabel('epochs')
        tloss_ax.set_ylabel('loss')
        tloss_ax.grid(True)
        tloss_ax.legend(loc='best')
        tloss_ax.set_title('Total Loss Across HAWQ Model')
        tloss_plt.tight_layout()
        tloss_plt.savefig(path.join(outputDir, filename))
        tloss_plt.show()

def plot_total_acc(model_set, model_acc_set, model_estop_set, outputDir='..'):
    # Total loss over fine tuning
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    for model, model_acc, model_estop in zip(model_set, model_acc_set, model_estop_set):
        tloss_plt = plt.figure()
        tloss_ax = tloss_plt.add_subplot()
        filename = 'total_acc_{}.png'.format(time)
        tloss_ax.plot(range(1, len(model_acc) + 1), model_acc, label='Test Accuracy')
        #tloss_ax.plot(range(1, len(model_loss[1]) + 1), model_loss[1], label='Validation Loss')
        # plot each stopping point
        for stop in model_estop:
            tloss_ax.axvline(stop+1, linestyle='--', color='r', alpha=0.3)
        tloss_ax.set_xlabel('epochs')
        tloss_ax.set_ylabel('Accuracy')
        tloss_ax.grid(True)
        tloss_ax.legend(loc='best')
        tloss_ax.set_title('Total Accuracy Across HAWQ Model Training')
        tloss_plt.tight_layout()
        tloss_plt.savefig(path.join(outputDir, filename))
        tloss_plt.show()