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
        self.rng = np.random.default_rng(seed=self.random_seed)

    def run(self, distribution_parameters):
        """Optimize target function with different kernels."""
        self.logs = defaultdict(list)
        seed_list = self.rng.integers(1e+9, size=self.num_tests)
        print(f'Optimizing {self.dim}-dimensional {self.function_name} function...')
        for t, seed in enumerate(seed_list):
            self.setup_target_function(seed)
            for dist_params in distribution_parameters:
                self.optimize(*dist_params, t)
        self.save_logs()

    def setup_target_function(self, random_seed):
        """Define target function to be minimized and sample an initial guess."""
        self.fun, self.x0 = setup_optimization(self.function_name, self.dim, random_seed)
        self.mc_rng = np.random.default_rng(seed=random_seed)

    def optimize(self, dist, params, t=0):
        """Minimize target function with a given distribution."""
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
        if dist.startswith('normal'):
            u = self.mc_rng.normal(size=(self.num_mc, self.dim))
            g = u / self.sigma
        elif dist.startswith('uniform'):
            u = 2*self.mc_rng.random(size=(self.num_mc, self.dim)) - 1
            g = u / self.sigma
        ##elif dist.startswith('cauchy'):
            ##u = self.mc_rng.standard_cauchy(size=(self.num_mc, self.dim))
            ##g = ??
        ##elif dist.startswith('laplace'):
            ##u = self.mc_rng.laplace(size=(self.num_mc, self.dim))
            ##g = ??
        elif dist.startswith('logistic'):
            u = self.mc_rng.logistic(size=(self.num_mc, self.dim))
            norm_u = np.linalg.norm(u, axis=1, keepdims=True)
            g = u / (norm_u * (1 + np.exp(-norm_u)))
        elif dist.startswith('t'):
            degree_of_freedom = 1
            u = self.mc_rng.standard_t(degree_of_freedom, size=(self.num_mc, self.dim))
            g = u * (degree_of_freedom + self.dim) / (degree_of_freedom + np.sum(u**2, axis=1))
        else:
            raise NameError(f'distribution {dist} is not recognized...')

        # estimate smoothed gradient
        fun_diff = (self.fun(x + self.sigma * u) - self.fun(x - self.sigma * u)).reshape(-1,1)
        grad = g * fun_diff
        return grad.mean(axis=0, keepdims=True)

    def save_logs(self):
        """Save logs to a csv file."""
        os.makedirs('./logs/', exist_ok=True)
        self.df = pd.DataFrame(exp.logs)
        self.df.to_csv(f'./logs/{self.exp_name}.csv')
        print(f'\n{self.df}')


if __name__ == '__main__':

    '''
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
                  'num_steps': 10000, 'num_mc': 100, 'num_tests': 10,
                  'random_seed': 0, 'exp_name': 'dev'}
    exp = Experiment(exp_params)
    exp.run(distribution_parameters)
    '''


    # hyperparameter search

    # setup experiment
    exp_params = {'function_name': 'sphere', 'dim': 100,
                  'num_steps': 10000, 'num_mc': 1000, 'num_tests': 10,
                  'random_seed': 0, 'exp_name': 'dev'}

    # define parameter grid
    funcs = ['sphere', 'ackley', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'schwefel']
    dists = ['normal', 'uniform', 'cauchy', 'laplace', 'logistic', 't']
    sigmas = [1e+1, 1e+0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    lrs = [1e+0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    # run search for all distributions and parameters
    for func in funcs:
        exp_params['function_name'] = func
        for dist in dists:
            exp_params['exp_name'] = f'{func}_{dist}'
            distribution_parameters = []

            # form parameter list
            for i, sigma in enumerate(sigmas):
                for j, lr in enumerate(lrs):
                    distribution_parameters.append([f'{dist}_{i}{j}', {'sigma': sigma, 'lr': lr}])

            # run experiment with these parameters
            exp = Experiment(exp_params)
            exp.run(distribution_parameters)

