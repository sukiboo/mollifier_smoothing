import os
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import truncnorm
from collections import defaultdict

from target_functions import setup_optimization


class Experiment:

    def __init__(self, experiment_parameters):
        self.__dict__.update(experiment_parameters)
        self.rng = np.random.default_rng(seed=self.random_seed)

    def run(self, distribution_parameters):
        """Optimize target function with different kernels."""
        self.logs = defaultdict(list)
        seed_list = self.rng.integers(1e+9, size=self.num_tests)
        print(f'Optimizing {self.dim}-dimensional {self.function_name} function...')
        for t, seed in enumerate(seed_list):
            self.setup_target_function(seed)
            for dist_params in distribution_parameters:
                self.optimize(*dist_params.items(), t)
        self.save_logs()

    def setup_target_function(self, random_seed):
        """Define target function to be minimized and sample an initial guess."""
        self.fun, self.x0 = setup_optimization(self.function_name, self.dim, random_seed)
        self.mc_rng = np.random.default_rng(seed=random_seed)

    def optimize(self, dist_params, t=0):
        """Minimize target function with a given distribution."""
        dist, params = dist_params
        savename = f'{dist}|{t}'
        self.__dict__.update(params)
        x = self.x0.copy()
        self.logs[savename].append(self.fun(x).item())
        for _ in tqdm(range(self.num_steps), desc=f'[{t+1}/{self.num_tests}] {dist:>8s}', ascii=True):
            grad = self.smoothed_gradient(x, dist)
            x -= self.lr * grad
            self.logs[savename].append(self.fun(x).item())

    def smoothed_gradient(self, x, dist):
        """Estimate smoothed gradient of the target function at the point x with a given kernel."""
        # sample perturbation vectors from a given distibution, see
        # https://numpy.org/doc/stable/reference/random/generator.html#distributions

        # normal distribution
        if dist.startswith('normal'):
            u = self.mc_rng.normal(size=(self.num_mc, self.dim))
            g = u

        # uniform distribution
        elif dist.startswith('uniform'):
            u = 2*self.mc_rng.random(size=(self.num_mc, self.dim)) - 1
            g = u

        # logistic distribution
        elif dist.startswith('logistic'):
            u = self.mc_rng.logistic(size=(self.num_mc, self.dim))
            norm_u = np.linalg.norm(u, axis=1, keepdims=True)
            g = 2*u / (1 + np.exp(-norm_u**2))

        # truncated normal distribution
        elif dist.startswith('truncnorm'):
            # this is wrong
            u = truncnorm.rvs(-1, 1, size=(self.num_mc, self.dim),
                              random_state=self.rng.integers(1e+9))
            g = u

        # t distribution with 2 degrees of freedom
        elif dist.startswith('t'):
            degrees_of_freedom = 2
            u = self.mc_rng.standard_t(degrees_of_freedom, size=(self.num_mc, self.dim))
            norm_u2 = np.sum(u**2, axis=1, keepdims=True)
            g = u * (degrees_of_freedom + self.dim) / (degrees_of_freedom + norm_u2)

        else:
            raise NameError(f'distribution {dist} is not recognized...')

        # estimate smoothed gradient
        fun_diff = (self.fun(x + self.sigma * u) - self.fun(x - self.sigma * u)).reshape(-1,1)
        grad = g * fun_diff / (2 * self.sigma)
        return grad.mean(axis=0, keepdims=True)

    def save_logs(self):
        """Save logs to a csv file."""
        os.makedirs('./logs/', exist_ok=True)
        self.df = pd.DataFrame(self.logs)
        self.df.to_csv(f'./logs/{self.exp_name}.csv')
        print(f'\n{self.df}')


if __name__ == '__main__':

    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='dev',
                        help='name of the config file in "./configs/"')

    # read the config file
    args = parser.parse_args()
    try:
        configs = yaml.safe_load(open(f'./configs/{args.config}.yml'))
    except:
        raise SystemExit(f'Could not read the file "./configs/{args.config}.yml".')

    # read parameters
    experiment_parameters = configs['experiment_parameters']
    experiment_parameters['exp_name'] = args.config
    distribution_parameters = configs['distribution_parameters']

    # setup and run the experiment
    exp = Experiment(experiment_parameters)
    exp.run(distribution_parameters)
