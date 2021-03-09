#!/usr/bin/env python
# coding: utf-8

# In[6]:


import multiprocessing
import sys
sys.path.append('../src')
from experiment_runner import *


# In[ ]:


# although these are the hyperparameters used in the paper, it is strongly recommended to use less blocks and smaller architectures 
n_blocks = 20
gamma = 5

dataset_params_dict = { 'cal_housing' : [8, [400,300,200,100], 'regression', 1e-5],
  'bike_sharing' : [15, [800,600,400,200], 'regression', 1e-5],
  'cifar' : [3072, [400,300,200,100], 'classification', 1e-5],
    'bank_marketing' : [48, [800,600,400,200], 'classification', 1e-5],
    'spambase' : [57, [800, 600, 400, 200], 'classification', 1e-5],
                        'skill' : [18, [300, 200, 100], 'classification', 1e-5]
}


# In[ ]:


def worker(inputs):
    """this runs a blocking call for dnn"""
    dataset = inputs[0]
    l0_lambda = inputs[1]
    maximum_order = inputs[2]
    dataset_params = inputs[3]
    
    print('starting dataset', dataset, 'max_order', maximum_order, 'l0', l0_lambda)
    run_experiment(dataset, l0_lambda, maximum_order, dataset_params, './', verbose=False, num_folds=5, waiting_fraction=40/dataset_params[0], n_blocks=n_blocks, gamma=gamma)
    return


# In[ ]:


# datasets = ['bike_sharing']
# datasets = ['bank_marketing']
datasets = ['skill']
l0_lambdas = [1e-2]
# l0_lambdas = [1e-2,1e-3,1e-4,1e-5]
jobs = []
if __name__ == '__main__':

    for dataset in datasets:
        if dataset == 'cifar': max_interaction_orders = [10,15,20]
        # else: max_interaction_orders = [2,3,4]
        else:
            # max_interaction_orders = [2,3,4]
            max_interaction_orders = [3,]
        for l0_lambda in l0_lambdas:
            for max_order in max_interaction_orders:
                dataset_params = dataset_params_dict[dataset]
                p = multiprocessing.Process(target=worker, args=((dataset, l0_lambda, max_order, dataset_params),))
                jobs.append(p)
                p.start()

