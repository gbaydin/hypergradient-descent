import numpy as np
import pandas as pd
import argparse
import csv
import os
import glob
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid.inset_locator import inset_axes

colorblindbright = [(252,145,100),(188,56,119),(114,27,166)]
colorblinddim    = [(213,167,103),(163,85,114),(104,59,130)]
for i in range(len(colorblindbright)):
    r, g, b = colorblindbright[i]
    colorblindbright[i] = (r / 255., g / 255., b / 255.)
for i in range(len(colorblinddim)):
    r, g, b = colorblinddim[i]
    colorblinddim[i] = (r / 255., g / 255., b / 255.)

colors = {'sgd':colorblinddim[0], 'sgdn':colorblinddim[1],'adam':colorblinddim[2], \
          'sgd_hd':colorblindbright[0], 'sgdn_hd':colorblindbright[1],'adam_hd':colorblindbright[2]}
names = {'sgd':'SGD','sgdn':'SGDN','adam':'Adam','sgd_hd':'SGD-HD','sgdn_hd':'SGDN-HD','adam_hd':'Adam-HD'}
linestyles = {'sgd':'--','sgdn':'--','adam':'--','sgd_hd':'-','sgdn_hd':'-','adam_hd':'-'}
linedashes = {'sgd':[3,3],'sgdn':[3,3],'adam':[3,3],'sgd_hd':[10,1e-9],'sgdn_hd':[10,1e-9],'adam_hd':[10,1e-9]}

parser = argparse.ArgumentParser(description='Plotting for hypergradient descent PyTorch tests', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', help='directory to read the csv files written by train.py', default='results', type=str)
opt = parser.parse_args()

for model in next(os.walk(opt.dir))[1]:
    data = {}
    data_epoch = {}
    selected = []
    for fname in glob.glob('{}/{}/**/*.csv'.format(opt.dir, model), recursive=True):
        name = os.path.splitext(os.path.basename(fname))[0]
        data[name] = pd.read_csv(fname)
        data_epoch[name] = data[name][np.isfinite(data[name].LossEpoch)]
        selected.append(name)

    plt.figure(figsize=(5,12))

    fig = plt.figure(figsize=(5, 12))
    ax = fig.add_subplot(311)
    for name in selected:
        plt.plot(data_epoch[name].Epoch,data_epoch[name].AlphaEpoch,label=names[name],color=colors[name],linestyle=linestyles[name],dashes=linedashes[name])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Learning rate')
    plt.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False, labelright=False)
    plt.grid()
    inset_axes(ax, width="50%", height="35%", loc=1)
    for name in selected:
        plt.plot(data[name].Iteration, data[name].Alpha,label=names[name],color=colors[name],linestyle=linestyles[name],dashes=linedashes[name])
    plt.yticks(np.arange(-0.01, 0.051, 0.01))
    plt.xlabel('Iteration')
    plt.ylabel('Learning rate')
    plt.xscale('log')
    plt.xlim([0,9000])
    plt.grid()

    ax = fig.add_subplot(312)
    for name in selected:
        plt.plot(data_epoch[name].Epoch, data_epoch[name].LossEpoch,label=names[name],color=colors[name],linestyle=linestyles[name],dashes=linedashes[name])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Training loss')
    plt.yscale('log')
    plt.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False, labelright=False)
    plt.grid()
    inset_axes(ax, width="50%", height="35%", loc=1)
    for name in selected:
        plt.plot(data[name].Iteration, data[name].Loss,label=names[name],color=colors[name],linestyle=linestyles[name],dashes=linedashes[name])
    plt.yticks(np.arange(0, 2.01, 0.5))
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.xscale('log')
    plt.xlim([0,9000])
    plt.grid()

    ax = fig.add_subplot(313)
    for name in selected:
        plt.plot(data_epoch[name].Epoch, data_epoch[name].ValidLossEpoch,label=names[name],color=colors[name],linestyle=linestyles[name],dashes=linedashes[name])
    plt.xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Validation loss')
    plt.yscale('log')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles,labels,loc='upper right',frameon=1,framealpha=1,edgecolor='black',fancybox=False)
    plt.grid()

    plt.tight_layout()
    plt.savefig('{}.pdf'.format(model), bbox_inches='tight')
