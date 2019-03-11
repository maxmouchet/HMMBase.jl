import time
import random
import numpy as np
import pandas as pd

import hmmlearn
from hmmlearn import hmm

from scipy.stats import dirichlet

def rand_init_distn(K):
    return dirichlet(np.ones(K)).rvs().ravel()

def rand_trans_matrix(K):
    return dirichlet(np.ones(K)).rvs(K)

def run_benchmark(Ks, Ts):
    print('Running benchmark...')
    print('Ks = {}'.format(Ks))
    print('Ts = {}'.format(Ts))

    results = []

    for K in Ks:
        for T in Ts:
            print('K = {}, T = {}'.format(K, T))
            
            π0 = rand_init_distn(K)
            π = rand_trans_matrix(K)

            model = hmm.GaussianHMM(n_components=K, covariance_type='full')
            model.startprob_ = π0
            model.transmat_ = π
            model.means_ = np.random.rand(K, 2)
            model.covars_ = np.tile(np.identity(2), (K, 1, 1))
            
            for i in range(20):
                start = time.process_time_ns()
                y, z = model.sample(T)
                time_ns = time.process_time_ns() - start
                results.append(('sample', 'np.array', K, T, time_ns))
            
            for i in range(20):
                start = time.process_time_ns()
                log_likelihoods = model._compute_log_likelihood(y)
                time_ns = time.process_time_ns() - start
                results.append(('compute_log_likelihood', 'np.array', K, T, time_ns))

            for i in range(20):
                start = time.process_time_ns()
                model._do_forward_pass(log_likelihoods)
                time_ns = time.process_time_ns() - start
                results.append(('do_forward_pass', 'np.array', K, T, time_ns))

            for i in range(20):
                start = time.process_time_ns()
                model._do_backward_pass(log_likelihoods)
                time_ns = time.process_time_ns() - start
                results.append(('do_backward_pass', 'np.array', K, T, time_ns))
                
            for i in range(20):
                start = time.process_time_ns()
                model._do_viterbi_pass(log_likelihoods)
                time_ns = time.process_time_ns() - start
                results.append(('do_viterbi_pass', 'np.array', K, T, time_ns))

    results = pd.DataFrame.from_records(results, columns=['fn', 'container', 'K', 'T', 'time_ns'])
    return results

if __name__ == '__main__':
    print('Loading benchmark...')

    seed = 2019
    random.seed(seed)
    np.random.seed(seed)

    version = hmmlearn.__version__
    print('hmmlearn v{}'.format(version))
    
    results = run_benchmark(range(2,22,2), [10, 100, 1000])
    out_fp = 'hmmlearn_v{}_{}_{}.csv'.format(version, int(time.time()), seed)
    
    print('Writing results to {}'.format(out_fp))
    results.to_csv(out_fp, index=False)
