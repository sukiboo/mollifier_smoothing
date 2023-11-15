import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=1.)


def visualize(logfile, percentile=(.25,.5,.75), show=True):
    '''Visualize optimization results.'''
    # read logs and aggregate statistics over multiple runs
    df = pd.read_csv(logfile, index_col=0)
    df_min = df.groupby(lambda x: x.split('|')[0], axis=1).quantile(percentile[0])
    df_med = df.groupby(lambda x: x.split('|')[0], axis=1).quantile(percentile[1])
    df_max = df.groupby(lambda x: x.split('|')[0], axis=1).quantile(percentile[2])

    # plot median and confidence interval for each algorithm
    fig, ax = plt.subplots(figsize=(8,5))
    algs = df_med.columns.to_list()
    for alg in algs:
        plt.plot(df_med[alg], linewidth=4, alpha=.9, label=alg)
        plt.fill_between(range(len(df_med)), df_min[alg], df_max[alg], alpha=.25)

    # configure axis
    ax.set_ylim(df.min().min(), df.loc[0].max())
    ax.set_title(f"{logfile.split('/')[-1].split('.')[0]}")
    plt.legend()
    plt.tight_layout()

    # save the results
    os.makedirs('./images/', exist_ok=True)
    savename = logfile.split('/')[-1].split('.')[0]
    plt.savefig(f'./images/{savename}.png', dpi=300, format='png')
    if show:
        plt.show()
    else:
        plt.close()


def visualize_search(logfile, percentile=(.25,.5,.75), show=True):
    '''Visualize results of hyperparameter search.'''
    # read logs and aggregate statistics over multiple runs
    df = pd.read_csv(logfile, index_col=0)
    df_min = df.groupby(lambda x: x.split('|')[0], axis=1).quantile(percentile[0])
    df_med = df.groupby(lambda x: x.split('|')[0], axis=1).quantile(percentile[1])
    df_max = df.groupby(lambda x: x.split('|')[0], axis=1).quantile(percentile[2])
    T = len(df_med)

    # plot median and confidence interval for each algorithm
    sns.set_palette(sns.color_palette(palette='hls', n_colors=56))
    fig, ax = plt.subplots(figsize=(12,6))
    algs = df_med.columns.to_list()
    for alg in algs:
        plt.plot(df_med[alg], linewidth=4, alpha=.9, label=f'{alg} ({df_med[alg][T-1]:.2e})')
        plt.fill_between(range(T), df_min[alg], df_max[alg], alpha=.25)

    # configure axis
    ax.set_ylim(df.min().min(), df.loc[0].max())
    ax.set_title(f"{logfile.split('/')[-1].split('.')[0]}")
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.8))

    # save the results
    os.makedirs('./images/search/', exist_ok=True)
    savename = logfile.split('/')[-1].split('.')[0]
    plt.savefig(f'./images/search/{savename}.png', dpi=300, format='png', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def find_best_parameters(logfile, percentile=.5):
    """Report the best hyperparameters."""
    # read logs and aggregate statistics over multiple runs
    df = pd.read_csv(logfile, index_col=0)
    df_stat = df.groupby(lambda x: x.split('|')[0], axis=1).quantile(percentile)
    T = len(df_stat)

    # find the parameters that provide the minimal values
    sigmas = [1e+1, 1e+0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    lrs = [1e+0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    best_ij = df_stat.loc[T-1].idxmin()
    i, j = int(best_ij[-2]), int(best_ij[-1])
    print(f'best parameters for {logfile}:  sigma = {sigmas[i]:.2e},  '\
        + f'lr = {lrs[j]:.2e},  median = {df_stat.loc[T-1][best_ij]:.2e}')


if __name__ == '__main__':

    # visualize each log file
    logdir = './logs/'
    for logfile in sorted(os.listdir(logdir)):
        find_best_parameters(logdir + logfile)
        ##visualize(logdir + logfile, show=True)
        ##visualize_search(logdir + logfile, show=False)

