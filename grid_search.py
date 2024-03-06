from experiment import Experiment


def grid_search(exp_params, num_steps):
    """Perform optimization with various parameter values."""
    funcs = ['sphere', 'ackley', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'schwefel']
    dists = ['normal', 'uniform', 'logistic', 't']
    sigmas = [1e+2, 3e+1, 1e+1, 3e+0, 1e+0, 3e-1, 1e-1, 3e-2,
              1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    lrs = [1e+1, 3e+0, 1e+0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3,
           1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]

    # run search for all distributions and parameters
    for func in funcs:
        exp_params['function_name'] = func
        for dist in dists:

            # modify experiment parameters for the current function and distribution
            exp_params['exp_name'] = f'{func}_{dist}'
            exp_params['num_steps'] = num_steps[func]
            distribution_parameters = []

            # form parameter list
            for i, sigma in enumerate(sigmas):
                for j, lr in enumerate(lrs):
                    distribution_parameters.append(
                        {f'{dist}_{i:02d}{j:02d}': {'sigma': sigma, 'lr': lr}})

            # run experiment with these parameters
            exp = Experiment(exp_params)
            exp.run(distribution_parameters)


if __name__ == '__main__':

    # experiment parameters
    exp_params = {'dim': 100, 'num_mc': 100, 'num_tests': 10, 'random_seed': 0}

    # number of iterations for each function
    num_steps = {'sphere': 1000,
                 'ackley': 25000,
                 'levy': 3000,
                 'michalewicz': 10000,
                 'rastrigin': 1000,
                 'rosenbrock': 1000,
                 'schwefel': 1000}

    grid_search(exp_params, num_steps)

