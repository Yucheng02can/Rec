# encoding = utf-8
import csv

import pandas as pd
import json
import os
import csv
import io
import sys
from tqdm import tqdm

import yaml
import datetime

""""
Searches all ‘result.log’ files in the results directory and returns the results sorted.
"""

log_dir = 'log'#'... /log' #log file
result_dir = 'result/temp_result'# The location where the results of the reading are stored.
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
select_headers = None  # ['nepochs','lr',‘AC’]  If the array is filled, only nepochs, lr, and AC will be read, and nothing else will be output.
remove_headers = ['device','checkpoint_path','models'] #Block ‘devices’, ‘checkpoint_path’, ‘models’ if found

def filter_dict(src_dict, select_filter, remove_filter):
    # Filtering data: Filtering out unwanted indicators
    tgt_dict = dict()

    # Forward filtering, retaining only the values whose key exists in the select_filter array.
    if select_headers is not None:
        for k in src_dict.keys():
            if k in select_headers:
                tgt_dict[k] = src_dict[k]
    else:
        tgt_dict = src_dict

    # Reverse filtering, removes all values whose keys exist in the array remove_filter.
    if remove_headers is not None:
        for k in remove_headers:
            if k in tgt_dict: tgt_dict.pop(k)

    return tgt_dict

def get_file_path(root_path, file_path_list):
    # Get the names of all files and directories in the directory
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # Get the path to a directory or file
        dir_file_path = os.path.join(root_path, dir_file)
        # Determine if the path is a file or a path
        if os.path.isdir(dir_file_path):
            # Recursively get the paths of all files and directories
            get_file_path(dir_file_path, file_path_list)
        else:
            file_path_list.append(dir_file_path)





def write_result_to_csv(result_dir,all_model_result):
    # pass
    #
    for file_name, model_results in tqdm(all_model_result.items(),total=len(all_model_result)):
        if not model_results : continue #Nothing comes of it. Nothing at all.
        filename = os.path.join(result_dir,file_name+".csv")
        # if os.path.isfile(filename):
        #     sys.stderr.writelines(f"{filename}The file already exists and the results of this model will no longer be summarised, if you need to summarise them, please delete and re-run them\n")
        #     continue
        df = pd.DataFrame(model_results)
        df.to_csv(filename, index=False, header=True)



def parse_result(lines,model_and_dataset):
    # models name
    data_name = model_and_dataset.split("_")[1]
    model_name = model_and_dataset.split("_")[0]

    # 结果
    for i in range(len(lines)):
        if lines[i].startswith('RESULTS='):
            result_str = lines[i].split("RESULTS=")[-1]
        if lines[i].startswith('GROUP_RESULTS='):
            cold_result_str = lines[i].split("GROUP_RESULTS=")[-1]  #Only the most recent results are retained.
        if lines[i].startswith('HEAD_RESULTS='):
            bias_result_str = lines[i].split("HEAD_RESULTS=")[-1]  #Only the most recent results are retained.
    if hasattr(yaml,'full_load'):
        result = yaml.full_load(io.StringIO(result_str))
        cold_result = yaml.full_load(io.StringIO(cold_result_str))
        bias_result = yaml.full_load(io.StringIO(bias_result_str))
    else:
        result = yaml.load(io.StringIO(result_str))
        cold_result = yaml.load(io.StringIO(cold_result_str))
        bias_result = yaml.load(io.StringIO(bias_result_str))
    #Filtering data
    # result = filter_dict(result, select_headers, remove_headers)
    # cold_result = filter_dict(cold_result, select_headers, remove_headers)
    # bias_result = filter_dict(bias_result, select_headers, remove_headers)

    cold_res = cold_bias_trans(cold_result)
    bias_res = cold_bias_trans(bias_result)

    result.update(cold_res)
    result.update(bias_res)
    # running time
    results = dict()
    results.update(result)


    return (model_name+"-"+data_name,results)

def cold_bias_trans(res:dict):
    bias_or_cold_res = {}
    for key, value in res.items():
        for m in value:
            metric_name=list(m.keys())[0]
            v=m[metric_name]
            f_key=key+"_"+metric_name
            bias_or_cold_res[f_key] = v
    return bias_or_cold_res


def scan(root_dir,model_name=''):
    #Recursive iteration catalogue, first level of catalogue is model name, second level of catalogue is runtime + parameter hash
    #A model outputs a table.
    # If model_name is specified, only one subdirectory is searched, otherwise all subdirectories are searched.

    if model_name:
        model_names = [model_name]
    else:
        model_names = os.listdir(root_dir)

    # Record the results of all model runs
    # {'dgcf':[{lr:0.01,ds:'ciao','ndcg10':0.00023},...,{{lr:0.001,ds:'ciao','ndcg10':0.013}}],
    # 'mhcn':[{lr:0.01,ds:'ciao','ndcg10':0.00023},...,{{lr:0.001,ds:'ciao','ndcg10':0.013}}]}
    all_result = {}
    for index,model_name in enumerate(model_names):
        log_dir = os.path.join(root_dir, model_name)
        file_path_list = []
        get_file_path(log_dir, file_path_list)
        # Record the results of all runs of the model, with each element of the list representing a run
        # [{lr:0.01,ds:'ciao','ndcg10':0.00023},...,{{lr:0.001,ds:'ciao','ndcg10':0.013}}]
        model_result = []
        print(model_names[index])
        for file_path in tqdm(file_path_list,total=len(file_path_list)):
            if not file_path.endswith('result.log'): continue
            with open(file_path, 'r') as fr:
                """logs: 
                    Train Start Time:2023-01-10_11-06-48 
                    ds:small
                    algorithm-parameter={'lr': 0.0005, 'n_iter': 2000, 'batch_size': 4096, 'channels_num': 4, 'iterations': 3, 'ui_layers': 1, 'uu_layers': 1, 'n_layers': 2, 'reg_weight': 0.0001, 'ind_weight': 0, 'recon_weight': 0, 'ssl_weight': 0, 'embedding_size': 64, 'eval_batch_user_size': 2000, 'full_item_predict': False, 'fuse_method': 'add', 'aux_loss': ['none'], 'topk': [5, 10, 15], 'use_cold_start_testset': False, 'cold_start_inter_num': [0, 20]}
                    RESULTS={'metrics.NDCG10': 0.44800741855504483, 'metrics.RECALL10': 1.0, 'metrics.Precision10': 0.125, 'metrics.MRR10': 0.2708333333333333, 'metrics.NDCG5': 0.44800741855504483, 'metrics.RECALL5': 1.0, 'metrics.Precision5': 0.2, 'metrics.NDCG15': 0.44800741855504483, 'metrics.RECALL15': 1.0, 'metrics.Precision15': 0.125}
                    Train Stop Time:2023-01-10 11:07:13
                    Time cost: 0.01 min
                """
                content = fr.readlines()
                # if "RESULTS" not in content[-2]: continue
                try:
                    key,values = parse_result(content,model_name)
                    #key=model_name-data_name, values={params,results}

                    if key not in all_result:
                        all_result[key]=[]
                    all_result[key].append(values)
                except:
                    print("error")
                    pass
    return all_result


if __name__ == '__main__':
    if not os.path.isdir(result_dir): os.makedirs(result_dir)
    models_result = scan(log_dir)
    write_result_to_csv(result_dir, models_result)