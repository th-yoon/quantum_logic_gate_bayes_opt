from unittest import result
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from util import save_data, surfcont
import numpy as np
import matlab.engine
import argparse
import sys
import time
import multiprocessing as mp
import os
import csv
import ray
from ray.util import inspect_serializability
from scipy import stats
from matplotlib import pyplot as plt
import pathlib

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def find_max(H, threshold, engine):
    """
    Finds the miminum time to satify the target threshold of fidelity,
    its corresponding list index, fidelity value, time grid, Bz_L, Bz_R, J, 
    and whether it exceeded the threshold (org_no_threshold, corr_no_threshold).

    Parameters:
    H (BayesianOptimization): BayesianOptimization object from bayes_opt package.
    threshold (float): target threshold of fidelity in [0, 1].

    Returns:
    result (dict): dictionary of the miminum time to satify the target threshold of fidelity,
    its corresponding list index, fidelity value, time grid, Bz_L, Bz_R, J, 
    and whether it exceeded the threshold (org_no_threshold, corr_no_threshold).
    """
    eng = engine
    [T, F_org, F_corr] = eng.CNOTcontrol_v1_201122_hyperparam(
        float(H['Bz_L']) * 1e9, float(H['Bz_R']) * 1e9, float(H['J']) * 1e5, nargout=3)
    T = np.array(T[0], dtype=float)
    F_org = np.ravel(np.array(F_org, dtype=float))
    F_corr = np.ravel(np.array(F_corr, dtype=float))

    T_org_best = len(T) - 1
    T_corr_best = len(T) - 1
    F_org_best = len(T) - 1
    F_corr_best = len(T) - 1
    org_no_threshold = True 
    corr_no_threshold = True 
    for i in range(len(F_org)):
        if F_org[i] > threshold:
            F_org_best = F_org[i]
            T_org_best = i
            org_no_threshold = False
            break
    for i in range(len(F_corr)):
        if F_corr[i] > threshold:
            F_corr_best = F_corr[i]
            T_corr_best = i
            corr_no_threshold = False
            break

    result = {
        'T': T, 
        'F_org': F_org, 
        'F_corr': F_corr, 
        'T_org_best': T_org_best,
        'T_corr_best': T_corr_best,
        'F_org_best': F_org_best,
        'F_corr_best': F_corr_best,
        'Bz_L': H['Bz_L'],
        'Bz_R': H['Bz_R'],
        'J': H['J'],
        'org_no_threshold': org_no_threshold,
        'corr_no_threshold': corr_no_threshold,
        }
    return result 

@ray.remote(num_cpus=1)
def find_hyperparam_min_t_mp(args, seed, target):
    np.random.seed(seed)
    H_optimizer = BayesianOptimization(
        f=None,
        pbounds={'Bz_L': (1, 10), 'Bz_R': (1, 10), 'J': (3, 1000)},
        verbose=2,
        random_state=seed,)
    H_utility = UtilityFunction(kind='ucb', kappa=10, xi=0.1)
    engine = matlab.engine.start_matlab()
    H_max = {
        'T': [], 
        'F_org': [],
        'F_corr': [],
        'T_org_best': [],
        'T_corr_best': [],
        'F_org_best': [], 
        'F_corr_best': [],
        'best_value': [],
        'Bz_L': [],
        'Bz_R': [],
        'J': [],
        'org_no_threshold': [],
        'corr_no_threshold': [],
        }
    F_corr_best_iteration = 0
    min_ts = []
    prev_best_T_corr = float('inf')
    engine = matlab.engine.start_matlab()
    for j in range(args.i):
        next_H = H_optimizer.suggest(H_utility)
        o_start = time.time()
        result = find_max(H=next_H, threshold=args.threshold, engine=engine)
        o_end = time.time()
        print('PID:', os.getpid(), '\n', j, ': elapsed time for optimization (s): ', o_end - o_start)
        H_optimizer.set_bounds(new_bounds={'Bz_L': (1, 10), 'Bz_R': (1, 10), 'J': (3, 1000)})
        print('next_{}:{}'.format(target, result[target]))
        print('next optimizer.max:{}'.format(H_optimizer.max))
        print('next_H:{}'.format(next_H))
        H_optimizer.register(
            params=next_H,
            target=0 - (result['T'][int(result[target])] * 1e9))
        for k in result:
            H_max[k].append(result[k])
        H_max['best_value'].append(H_optimizer.max)
        if H_max['T'][-1].reshape(-1, 1)[H_max['T_corr_best'][-1]] < prev_best_T_corr:
            F_corr_best_iteration = len(H_max['T']) - 1 
            prev_best_T_corr = H_max['T'][-1].reshape(-1, 1)[H_max['T_corr_best'][-1]]
        min_ts.append(-H_optimizer.max['target']/1e9)
    return min_ts

def find_hyperparam(H_optimizer, H_utility, args, target):
    """
    1) Gets the next parameters to try suggested from the utility function.
    2) Evaluate the function with the suggested parameters and gets the sampled target value.
    3) Register the sampled target value and its corresponding parameters.
    4) Repeat.

    Parameters:
    H_optimizer (BayesianOptimization): BayesianOptimization object from bayes_opt package.
    H_utility (UtilityFunction): UtilityFunction object from bayes_opt package.
    args (dict): argument dictionary.
    target (float): the target to maximize - in this case, minimum time to satisfy the fidelity threshold.

    Returns:
    H_max (dict): final H_optimizer.max after evaluations.
    H_optimizer (BayesianOptimization): final BayesianOptimization object after evaluations.
    
    """ 
    H_optimizer = H_optimizer[0]
    H_max = {
        'T': [], 
        'F_org': [],
        'F_corr': [],
        'T_org_best': [],
        'T_corr_best': [],
        'F_org_best': [], 
        'F_corr_best': [],
        'best_value': [],
        'Bz_L': [],
        'Bz_R': [],
        'J': [],
        'org_no_threshold': [],
        'corr_no_threshold': [],
        }
    F_corr_best_iteration = 0
    prev_best_T_corr = float('inf')
    engine = matlab.engine.start_matlab()
    for j in range(args.i):
        next_H = H_optimizer.suggest(H_utility)
        o_start = time.time()
        result = find_max(H=next_H, threshold=args.threshold, engine=engine)
        o_end = time.time()
        print('PID:', os.getpid(), '\n', j, ': elapsed time for optimization (s): ', o_end - o_start)
        H_optimizer.set_bounds(new_bounds={'Bz_L': (1, 10), 'Bz_R': (1, 10), 'J': (3, 1000)})
        print('next_{}:{}'.format(target, result[target]))
        print('next optimizer.max:{}'.format(H_optimizer.max))
        print('next_H:{}'.format(next_H))
        H_optimizer.register(
            params=next_H,
            target=0 - (result['T'][int(result[target])] * 1e9))
        for k in result:
            H_max[k].append(result[k])
        H_max['best_value'].append(H_optimizer.max)
        if H_max['T'][-1].reshape(-1, 1)[H_max['T_corr_best'][-1]] < prev_best_T_corr:
            F_corr_best_iteration = len(H_max['T']) - 1 
            prev_best_T_corr = H_max['T'][-1].reshape(-1, 1)[H_max['T_corr_best'][-1]]
        save_data(
            H_max,
            F_corr_best_iteration,
            f'corr',
            args.threshold,
            'plot_data/',
            j,
        )
        if j > 2:
            surfcont('plot_data/', f'corr_{args.threshold}_{j}_i={j-1}.csv', H_optimizer, H_utility)
    np.save(f'data/best_t_i={args.i}_th={args.threshold}.npy', H_max['T_corr_best'])
    return [H_max, H_optimizer]

def main():
    parser = argparse.ArgumentParser('cnot_control')
    parser.add_argument(
        '-i', help='Number of outer iterations (default: 50)', default=200,
        type=int)
    parser.add_argument(
        '--threshold', help='Fidelity target threshold', default=0.9,
        type=float)
    parser.add_argument(
        '--seed', help='seed', default=9999999,
        type=int)
    parser.add_argument(
        '--n_workers', help='The number of parallel workers', default=10,
        type=int)
    parser.add_argument(
        '--min_t_plot', help='Plot min_t over multiple workers?', default=False,
        type=str2bool)
    args = parser.parse_args()
    pathlib.Path('min_t/').mkdir(exist_ok=True)
    pathlib.Path('plot_data/').mkdir(exist_ok=True)
    pathlib.Path('data/').mkdir(exist_ok=True)

    t_start = time.time()
    if args.min_t_plot:
        inspect_serializability(find_hyperparam_min_t_mp, name='find_hyperparam')
        ray.init(ignore_reinit_error=True, num_cpus=int(os.cpu_count()), num_gpus=0)
        result_ids = []
        for seed in range(args.n_workers):
            result_ids.append(
                find_hyperparam_min_t_mp.remote(args, seed+10000, 'T_corr_best')
                )
        min_ts = np.array(ray.get(result_ids))
        mu_min_ts = np.mean(min_ts, axis=0)
        sigma_min_ts = np.std(min_ts, axis=0)
        np.save(f'min_t/min_ts_i={args.i}_workers={args.n_workers}_th={args.threshold}.npy', min_ts)
        fig, ax = plt.subplots()
        steps = np.arange(min_ts.shape[1])
        ax.plot(steps, mu_min_ts)
        ax.fill_between(steps, (mu_min_ts-1.96*sigma_min_ts/np.sqrt(min_ts.shape[0])), (mu_min_ts+1.96*sigma_min_ts/np.sqrt(min_ts.shape[0])), color='b', alpha=0.1)
        ax.set_xlabel('steps')
        ax.set_ylabel(r'$t_{CNOT}$')
        fig.savefig(f'min_t/min_ts_i={args.i}_workers={args.n_workers}_th={args.threshold}.png')
    else:
        H_optimizer = BayesianOptimization(
            f=None,
            pbounds={'Bz_L': (1, 10), 'Bz_R': (1, 10), 'J': (3, 1000)},
            verbose=2,
            random_state=args.seed,)
        H_utility = UtilityFunction(kind='ucb', kappa=10, xi=0.1)

        ret_corr = find_hyperparam([H_optimizer], H_utility, args, 'T_corr_best')
        ret_corr_get = ret_corr
        print('total elapsed time (s): ', t_end - t_start)

        F_corr_best_iteration = np.argmin(ret_corr_get[0]['T_corr_best'])
        save_data(
            ret_corr_get[0]['F_corr'][F_corr_best_iteration],
            ret_corr_get[1].res,
            ret_corr_get[0],
            F_corr_best_iteration,
            f'corr',
            args.threshold,
            'plot_data/'
        )

        print('----F_corr parameters----')
        print('Bz_L:', ret_corr_get[0]['Bz_L'][F_corr_best_iteration])
        print('Bz_R:', ret_corr_get[0]['Bz_R'][F_corr_best_iteration])
        print('J:', ret_corr_get[0]['J'][F_corr_best_iteration])
    t_end = time.time()

if __name__ == "__main__":
    main()