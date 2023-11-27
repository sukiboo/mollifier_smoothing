"""
    A list of optimization benchmark functions from
    https://www.sfu.ca/~ssurjano/optimization.html

    Each target function should take a rank 2 numpy array as an input,
    i.e. x.shape = (n, dim), where n is the batch size and dim is the
    space dimensionality.
"""

import numpy as np
try:
    import numexpr as ne
    use_numexpr = True
except:
    print('Not using numexpr, functions are going to be evaluated with numpy.')
    use_numexpr = False


def target_function(function_name, dim):
    """Setup benchmark function."""
    # Ackley function
    if function_name == 'ackley':
        if use_numexpr:
            def fun(x):
                a = 20
                b = .2
                c = 2*np.pi
                mean1 = np.mean(x**2, axis=1)
                mean2 = np.mean(ne.evaluate('cos(c*x)'), axis=1)
                val = ne.evaluate('-a * exp(-b * mean1**.5) - exp(mean2) + a + exp(1)')
                return val
        else:
            a = 20
            b = .2
            c = 2*np.pi
            fun = lambda x: -a * np.exp(-b * np.sqrt(np.mean(x**2, axis=1)))\
                - np.exp(np.mean(np.cos(c*x), axis=1)) + a + np.exp(1)
        x_dom = [[-32.768, 32.768]] * dim
        x_min = [0] * dim

    # Griewank function
    elif function_name == 'griewank':
        if use_numexpr:
            def fun(x):
                weights = 1 + np.arange(dim)
                sum1 = np.sum(ne.evaluate('x**2'), axis=1)
                prod1 = np.prod(ne.evaluate('cos(x / weights**.5)'), axis=1)
                val = sum1 / 4000 - prod1 + 1
                return val
        else:
            weights = 1 + np.arange(dim)
            fun = lambda x: np.sum(x**2, axis=1) / 4000\
                - np.prod(np.cos(x/np.sqrt(weights)), axis=1) + 1
        x_dom = [[-600, 600]] * dim
        x_min = [0] * dim

    # Levy function
    elif function_name == 'levy':
        if use_numexpr:
            def fun(x):
                w = (x.T + 3) / 4
                w1 = w[0]
                wi = w[:-1]
                wd = w[-1]
                pi = np.pi
                first = ne.evaluate('sin(pi*w1)**2')
                mid = np.sum(ne.evaluate('(wi - 1)**2 * (1 + 10*sin(pi*wi + 1)**2)'), axis=0)
                last = ne.evaluate('(wd - 1)**2 * (1 + sin(2*pi*wd)**2)')
                val = first + mid + last
                return val
        else:
            w = lambda x: (x + 3) / 4
            fun = lambda x: np.sin(np.pi*w(x)[:,0])**2\
                + np.sum((w(x)[:,:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w(x)[:,:-1] + 1)**2), axis=1)\
                + (w(x)[:,-1] - 1)**2 * (1 + np.sin(2*np.pi*w(x)[:,-1])**2)
        x_dom = [[-10, 10]] * dim
        x_min = [1] * dim

    # Michalewicz function
    elif function_name == 'michalewicz':
        if use_numexpr:
            def fun(x):
                m = 10
                pi = np.pi
                weights = 1 + np.arange(dim)
                val = -np.sum(ne.evaluate(f'sin(x) * sin(weights * x**2 / pi)**(2*m)'), axis=1)
                return val
        else:
            m = 10
            weights = 1 + np.arange(dim)
            fun = lambda x: -np.sum(np.sin(x) * np.sin(weights * x**2 / np.pi)**(2*m), axis=1)
        x_dom = [[0, np.pi]] * dim
        x_min = [None] * dim

    # Rastrigin function
    elif function_name == 'rastrigin':
        if use_numexpr:
            def fun(x):
                pi = np.pi
                val = 10*dim + np.sum(ne.evaluate('x**2 - 10*cos(2*pi*x)'), axis=1)
                return val
        else:
            fun = lambda x: 10*dim + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
        x_dom = [[-5.12, 5.12]] * dim
        x_min = [0] * dim

    # Rosenbrock function
    elif function_name == 'rosenbrock':
        if use_numexpr:
            def fun(x):
                xj, xi = x[:,1:], x[:,:-1]
                val = np.sum(ne.evaluate('100*(xj - xi**2)**2 + (xi - 1)**2'), axis=1)
                return val
        else:
            fun = lambda x: np.sum(100 * (x[:,1:] - x[:,:-1]**2)**2 + (x[:,:-1] - 1)**2, axis=1)
        x_dom = [[-5, 10]] * dim
        x_min = [1] * dim

    # Schwefel function -- taking absolute value to ensure the global minimum existance
    elif function_name == 'schwefel':
        if use_numexpr:
            def fun(x):
                val = np.abs(418.9829*dim - np.sum(ne.evaluate('x * sin(abs(x)**.5)'), axis=1))
                return val
        else:
            fun = lambda x: np.abs(418.9829*dim - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1))
        x_dom = [[-500, 500]] * dim
        x_min = [420.9687] * dim

    # sphere function
    elif function_name == 'sphere':
        if use_numexpr:
            def fun(x):
                val = np.sum(ne.evaluate('x**2'), axis=1)
                return val
        else:
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

