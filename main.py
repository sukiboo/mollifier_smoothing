import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from target_functions import setup_optimization


#TODO: visualization
class Experiment:

    def __init__(self, exp_params):
        self.__dict__.update(exp_params)
        self.setup_target_function()

    def setup_target_function(self):
        """Define target function to be minimized and sample an initial guess."""
        self.fun, self.x0 = setup_optimization(self.function_name, self.dim, self.random_seed)
        self.logs = defaultdict(list)
        self.mc_rng = np.random.default_rng(seed=self.random_seed)

    def run(self, distribution_parameters):
        """Optimize target function with different kernels."""
        print(f'Optimizing {self.dim}-dimensional {self.function_name} function...')
        for dist_params in distribution_parameters:
            self.optimize(*dist_params)
        self.save_logs()

    def optimize(self, dist, params):
        """Minimize target function with a given distribution."""
        self.__dict__.update(params)
        x = self.x0.copy()
        self.logs[dist].append(self.fun(x))
        for _ in tqdm(range(self.num_steps), desc=f'{dist:>8s}'):
            grad = self.smoothed_gradient(x, dist)
            x -= self.lr * grad
            self.logs[dist].append(self.fun(x))

    def smoothed_gradient(self, x, dist):
        """Estimate smoothed gradient of the target function at the point x with a given kernel."""
        # sample perturbation vectors from a given distibution, see
        # https://numpy.org/doc/stable/reference/random/generator.html#distributions
        if dist == 'normal':
            u = self.mc_rng.normal(size=(self.num_mc, self.dim))
        elif dist == 'uniform':
            u = 2*self.mc_rng.random(size=(self.num_mc, self.dim)) - 1
        elif dist == 'cauchy':
            u = self.mc_rng.standard_cauchy(size=(self.num_mc, self.dim))
        elif dist == 'laplace':
            u = self.mc_rng.laplace(size=(self.num_mc, self.dim))
        elif dist == 'logistic':
            u = self.mc_rng.logistic(size=(self.num_mc, self.dim))
        elif dist == 't':
            u = self.mc_rng.standard_t(1, size=(self.num_mc, self.dim))
        else:
            raise NameError(f'distribution {dist} is not recognized...')

        # estimate smoothed gradient
        grad = u * (self.fun(x + self.sigma * u) - self.fun(x - self.sigma * u)) / (2 * self.sigma)
        return grad.mean(axis=0)

    def save_logs(self):
        """Save logs to a csv file."""
        os.makedirs('./logs/', exist_ok=True)
        self.df = pd.DataFrame(exp.logs)
        self.df.to_csv(f'./logs/{self.exp_name}.csv')
        print(f'\n{self.df}')


if __name__ == '__main__':

    # setup distribution parameters
    distribution_parameters = [
        ['normal', {'sigma': 1., 'lr': 1e-3}],
        ['uniform', {'sigma': 1., 'lr': 1e-3}],
        ['cauchy', {'sigma': 1., 'lr': 1e-3}],
        ['laplace', {'sigma': 1., 'lr': 1e-3}],
        ['logistic', {'sigma': 1., 'lr': 1e-3}],
        ['t', {'sigma': 1., 'lr': 1e-3}],
        ]

    # setup experiment
    exp_params = {'function_name': 'sphere', 'dim': 100,
                  'num_steps': 10000, 'num_mc': 100,
                  'random_seed': 0, 'exp_name': 'test'}
    exp = Experiment(exp_params)
    exp.run(distribution_parameters)

