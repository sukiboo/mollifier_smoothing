"""
    A list of optimization benchmark functions from
    https://www.sfu.ca/~ssurjano/optimization.html

    Each target function should take a rank 2 numpy array as an input,
    i.e. x.shape = (n, dim), where n is the batch size and dim is the
    space dimensionality.
"""

import numpy as np


def target_function(function_name, dim):
    """Setup benchmark function."""
    # Ackley function
    if function_name == 'ackley':
        fun = lambda x: -20 * np.exp(-.2 * np.sqrt(np.mean(x**2, axis=1)))\
            - np.exp(np.mean(np.cos(2*np.pi*x), axis=1)) + 20 + np.exp(1)
        x_dom = [[-32.768, 32.768]] * dim
        x_min = [0] * dim

    # Griewank function
    elif function_name == 'griewank':
        fun = lambda x: np.sum(x**2, axis=1)/4000\
            - np.prod(np.cos(x/np.sqrt(np.arange(dim)+1)), axis=1) + 1
        x_dom = [[-600, 600]] * dim
        x_min = [0] * dim

    # Levy function
    elif function_name == 'levy':
        w = lambda x: (x + 3) / 4
        fun = lambda x: np.sin(np.pi*w(x)[:,0])**2\
            + np.sum((w(x)[:,:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w(x)[:,:-1] + 1)**2), axis=1)\
            + (w(x)[:,-1] - 1)**2 * (1 + np.sin(2*np.pi*w(x)[:,-1])**2)
        x_dom = [[-10, 10]] * dim
        x_min = [1] * dim

    # Michalewicz function
    elif function_name == 'michalewicz':
        m = 10 # default parameter
        fun = lambda x: -np.sum(np.sin(x) * np.sin(np.arange(1,dim+1) * x**2 / np.pi)**(2*m), axis=1)
        x_dom = [[0, np.pi]] * dim
        x_min = [None] * dim

    # Rastrigin function
    elif function_name == 'rastrigin':
        fun = lambda x: 10*dim + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
        x_dom = [[-5.12, 5.12]] * dim
        x_min = [0] * dim

    # Rosenbrock function
    elif function_name == 'rosenbrock':
        fun = lambda x: np.sum(100 * (x[:,1:] - x[:,:-1]**2)**2 + (x[:,:-1] - 1)**2, axis=1)
        x_dom = [[-5, 10]] * dim
        x_min = [1] * dim

    # Schwefel function -- taking absolute value to ensure the global minimum existance
    elif function_name == 'schwefel':
        fun = lambda x: np.abs(418.9829*dim - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1))
        x_dom = [[-500, 500]] * dim
        x_min = [420.9687] * dim

    # sphere function
    elif function_name == 'sphere':
        fun = lambda x: np.sum(x**2, axis=1)
        x_dom = [[-5.12, 5.12]] * dim
        x_min = [0] * dim

    else:
        raise SystemExit(f'function {function_name} is not defined...')

    return fun, np.array(x_min, ndmin=2), np.array(x_dom).T


def initial_guess(x_dom, random_seed):
    """Randomly sample initial guess"""
    np.random.seed(random_seed)
    dim = x_dom.shape[-1]
    x0 = (x_dom[1] - x_dom[0]) * np.random.rand(dim) + x_dom[0]
    return np.array(x0, ndmin=2)


def setup_optimization(function_name, dim, random_seed=0, noise=0):
    """Return a target function and an initial guess"""
    target_fun, x_min, x_dom = target_function(function_name, dim)
    x0 = initial_guess(x_dom, random_seed)

    # add random noise
    noise_fun = lambda x: noise * (2*np.random.rand(x.shape[0]) - 1)
    fun = lambda x: target_fun(x) * (1 + noise_fun(x))

    return fun, x0

