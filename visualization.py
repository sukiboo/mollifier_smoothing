import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=.8)


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
    sigmas = [1e+2, 3e+1, 1e+1, 3e+0, 1e+0, 3e-1, 1e-1, 3e-2,
              1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    lrs = [1e+1, 3e+0, 1e+0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3,
           1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    best_ij = df_stat.loc[T-1].idxmin()
    i, j = int(best_ij[-4:-2]), int(best_ij[-2:])
    print(f'best parameters for {logfile}:  sigma = {sigmas[i]:.2e},  '\
        + f'lr = {lrs[j]:.2e},  median = {df_stat.loc[T-1][best_ij]:.2e}')


def plot_hyperparameter_heatmap(percentile=.5, dists=['logistic', 'normal', 't', 'uniform'],
        ##funcs=['ackley', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'schwefel']):
        funcs=['schwefel']):
    """Plot grid searches for each function and distribution."""
    # grid search parameters
    sigmas = [1e+2, 3e+1, 1e+1, 3e+0, 1e+0, 3e-1, 1e-1, 3e-2,
              1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    lrs = [1e+1, 3e+0, 1e+0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3,
           1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    val_range = {'ackley': (0, 25), 'levy': (1000, 1500),
                 'michalewicz': (-30, 0), 'rastrigin': (0, 2000),
                 'rosenbrock': (0, 1.5e+7), 'schwefel': (0, 45000)}

    ##from mpl_toolkits.axes_grid1 import make_axes_locatable

    # plot grid for every function
    for func in funcs:
        fig, axs = plt.subplots(1, 4, figsize=(20,5))
        fig.suptitle(f'100d {func} function')

        # set up unified colorbar
        vmin, vmax = val_range[func]
        vmax = pd.read_csv(f'./logs/search/{func}_t.csv', index_col=0).loc[0].values.mean()
        print(vmax)
        cmap = cm.get_cmap('viridis_r')
        normalizer = Normalize(vmin, vmax)
        ##normalizer = LogNorm(vmin, vmax)
        cbar = cm.ScalarMappable(norm=normalizer, cmap=cmap)

        for dist, ax in zip(dists, axs):

            # read logs and aggregate statistics over multiple runs
            df = pd.read_csv(f'./logs/search/{func}_{dist}.csv', index_col=0)
            df = df.clip(upper=vmax)
            ##df_stat = df.groupby(lambda x: x.split('|')[0], axis=1).quantile(percentile)
            ##df_agg = df.T.groupby(lambda x: x.split('|')[0])
            df_agg = df.groupby(lambda x: x.split('|')[0], axis=1)
            ##df_stat = df_agg.median()
            ##df_stat = (df_agg.sum() - df_agg.min() - df_agg.max()) / 3
            df_stat = (df_agg.sum() - df_agg.min() - df_agg.max()) / 8

            # plot the quality matrix
            vals = df_stat.loc[len(df_stat)-1].values.reshape(len(sigmas), len(lrs))
            ##ax.imshow(vals, cmap=cmap, norm=Normalize(vmin=np.min(vals), vmax=np.percentile(vals, .5)))
            ax.imshow(vals, cmap=cmap, norm=normalizer)

            # highlight the best parameters
            best_slr = df_stat.loc[len(df_stat)-1].idxmin()
            s, lr = int(best_slr[-4:-2]), int(best_slr[-2:])
            ax.add_patch(plt.Rectangle((lr-.5, s-.5), 1, 1, color='r', linewidth=2, fill=False))

            # configure axis
            ax.set_title(f'{dist} distribution')
            ax.set_xlabel('learning rate')
            ax.set_ylabel('sigma')
            ax.set_xticks(np.arange(len(lrs)))
            ax.set_yticks(np.arange(len(sigmas)))
            ax.set_xticklabels(map(lambda x: f'{x:.2e}', lrs), rotation=45, ha='right')
            ax.set_yticklabels(map(lambda x: f'{x:.2e}', sigmas))
            ax.set_xticks(np.arange(len(lrs)) + .5, minor=True)
            ax.set_yticks(np.arange(len(sigmas)) + .5, minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
            ax.grid(which='major') # no idea why this disables major grid lines

        ##for i in range(grid_size[0]):
            ##for j in range(grid_size[1]):
                ##ax.text(j, i, f'{mu_mat[i,j]:.3f}', size=8, weight='bold', ha='center', va='center')

        # add colorbar
        ##divider = make_axes_locatable(fig)
        ##cax = divider.append_axes('right', size='20%', pad=1)#, aspect=10)
        ##cax = fig.add_axes([.95, .1, .02, .8])
        ##im = ax.imshow(data, cmap='bone')
        ##fig.colorbar(im, cax=cax, orientation='vertical')
        ##fig.colorbar(cbar, cax=cax)#, pad=.2, aspect=10)#, orientation='vertical')
        fig.colorbar(cbar, ax=axs[-1])#, pad=.2, aspect=10)


        # save the figure
        plt.tight_layout()
        os.makedirs('./images/grids', exist_ok=True)
        plt.savefig(f'./images/grids/{func}_grids.png', format='png', dpi=300)
        ##plt.show()
        plt.close()


if __name__ == '__main__':

    ##df = plot_hyperparameter_heatmap()

    ##'''
    # visualize each log file
    logdir = './logs/'
    ##logdir = './logs/search/'
    for logfile in sorted(os.listdir(logdir)):
        ##find_best_parameters(logdir + logfile)
        visualize(logdir + logfile, show=True)
    ##'''
