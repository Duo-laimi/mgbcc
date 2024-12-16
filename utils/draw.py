import torch
import random
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def draw_converge(y, fid, title, xlabel, ylabel, flag=False):
    n = len(y)
    x = list(range(0, n))
    plt.figure(fid)
    plt.plot(x, y)
    if flag:
        plt.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
    else:
        plt.ylim(0,1)
    plt.grid(True, linestyle=":")
    # plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


