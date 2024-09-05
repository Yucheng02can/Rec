import copy

import importlib
import logging
import os
import shutil

import pandas as pd
import scipy.sparse as sp
import numpy as np
import importlib
import sys
import torch
#-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
#----------------------------

def print_progress(p):
    print("current progress is %2.2f%%"%(p*100.0),file=sys.stderr)

filedict = {"algorithm":'algorithm.algorithm.PMF',
            "algorithm_parameters":{'d':20,'lr':0.1,'n_itr':500,'eps':0.00001},
            "protocol":'protocol01.Kfold5',
            "data_dir":'data',
            "metrics":['metric1.MAE','metric2.RMSE']}



def my_import(name):
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)
    return cls



def run(filedict=filedict,call_back=lambda x:None):


    assert "algorithm" in filedict
    assert "protocol" in filedict
    assert "data" in filedict
    assert "metrics" in filedict

    dataset_name = filedict['data']
    ds = parse_data(dataset_name)

    # log
    alg=filedict['algorithm'].split('.')[-1]
    out_path=f'log/{alg}_{dataset_name}'
    if not os.path.isdir(out_path):
        os.makedirs(out_path) #Creating Folders
    out_path = os.path.join(out_path,f"result.log")
    if not os.path.isfile(out_path):
        os.path
    # Creating Logger Objects
    logger = logging.getLogger(f'{alg}_{dataset_name}_logger')
    logger.setLevel(logging.DEBUG)  # Set the minimum log level to DEBUG

    # Create a file processor to write log messages to a file
    file_handler = logging.FileHandler(out_path)
    file_handler.setLevel(logging.DEBUG)  # Set the minimum logging level for the processor

    # Defining the Log Format
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)  # Applying formats to file processors

    # Adding a file handler to a Logger object
    logger.addHandler(file_handler)
    #Get all objects
    algorithm = my_import(filedict['algorithm'])()
    protocol = my_import(filedict['protocol'])()
    metrics = []
    for m in filedict['metrics']:
        metrics.append(my_import(m)())

    # Setting Algorithm Parameters
    for k,v in filedict['algorithm_parameters'].items():
        setattr(algorithm,k,v)

    results = protocol.test(algorithm,ds,metrics,dataset_name,logger,call_back)
    assert len(results)==len(metrics)

    return results# Returns the measurements for each indicator.


def parse_data(dataset_name):
    fdata = os.path.join("../dataset/",f'{dataset_name}.csv')
    # fdata = os.path.join(data_dir, 'rating.data')
    # df = pd.read_table(fdata,header=None,delimiter='\\s')
    df = pd.read_csv(fdata,header=None,engine='python', dtype={
    '0': 'int',
    '1': 'int',
    '2': 'float'
    })
    # if df.shape[1] == 1:
    #     df = pd.read_csv(fdata, header=None, delimiter='\\s',engine='python')
    uids = df.iloc[:, 0].values
    iids = df.iloc[:, 1].values
    rates = df.iloc[:, 2].values
    rates=[float(r) for r in rates]
    ds = sp.csr_matrix((rates, (uids, iids)),dtype=float)  # refer to ： csr_matrix((data,  (row_ind, col_ind)), [shape=(M, N)])
    return ds

if __name__ == "__main__":
    # The call requires custom filedict and print_progress variables.
    # filedic is used to configure algorithms, algorithm parameters, data, test criteria, etc.
    # The callback function for printing the current progress, print_progress.
    results = run(filedict,print_progress)
    print(results)
