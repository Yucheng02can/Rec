from scenario import run
from time import *
import json
import argparse
import torch

def smart_convert(value):
    assert  isinstance(value,str)
    if value.count('.')>0:
        try:
            return float(value)
        except:
            pass
    try : # Checking integers
        return int(value)
    except:
        return value


def myloads(jstr):

    jstr = jstr.replace('{','').replace('}','').strip()
    if len(jstr)==0:
        return {}
    tokens = jstr.split(',')
    tokens = [tk.split(':')  for tk in tokens]

    return dict((k,smart_convert(v)) for k,v in tokens)


parser = argparse.ArgumentParser(description='Algorithm test procedure')
parser.add_argument('-p',dest='protocol',  type=str,required=False,
                    default="rs_tvt.RSImplicitTVT",
                    help='Validation Methods')
parser.add_argument('-m',dest='metrics',  type=str,required=False,
                    default="metrics.NDCG5,metrics.NDCG10,metrics.NDCG20,metrics.RECALL5,metrics.RECALL10,metrics.RECALL20,metrics.MRR5,metrics.MRR10,metrics.MRR20",
                    help='List of validation metrics, comma-separated')
parser.add_argument('-a',dest='alg',  type=str,required=False,
                    default="Debias.algorithm.mf.MF",
                    help='Algorithms to be validated')
parser.add_argument('-d',dest='data',  type=str,required=False,
                    default="ml-100k",
                    help='data catalogue')
parser.add_argument('-r',dest='params',  type=myloads,required=False,
                    default="{}",
                    help='''algorithm parameter, which is a dictionary, e.g."{d:20,lr:0.1,n_itr:1000}" ''')


argparse

# Modify the following function to write progress p to \task_id\progress file
def print_progress(p):
    print("current progress is %2.2f%%"%(p*100.0))



if __name__ == '__main__':
    protocol="rs_tvt.RSImplicitTVT"
    metrics="metrics.NDCG5,metrics.NDCG10,metrics.NDCG20,metrics.RECALL5,metrics.RECALL10,metrics.RECALL20,metrics.MRR5,metrics.MRR10,metrics.MRR20"

    filedict = {"algorithm": "Debias.algorithm.mf.MF",
         "protocol": protocol,
        "algorithm_parameters": {},
         "data": "",
         "metrics": metrics.split(',')}

    data_list=['ml-100k','ml-1m','lastfm','Book-Crossing']
    alg_list=["Debias.algorithm.engcn.EnGCN","Debias.algorithm.lightgcn.LightGCN","Debias.algorithm.mf.MF","Debias.algorithm.ngcf.NGCF","Debias.algorithm.neumf.NeuMF"]
    alg_list=["algorithm.engcn.EnGCN","algorithm.lightgcn.LightGCN","algorithm.mf.MF","algorithm.ngcf.NGCF","algorithm.neumf.NeuMF"]
    for data in data_list:
        for alg in alg_list:
            print(data,alg.split(".")[-1])
            filedict["data"]=data
            filedict["algorithm"]=alg
            results = run(filedict,print_progress)
            print("Final Results is :",results,filedict)

    # # The following is read from the database
    # filedict = {"algorithm": args.alg,
    #             "algorithm_parameters": args.params,
    #             "protocol": args.protocol,
    #             "data": args.data,
    #             "metrics": args.metrics.split(',')}
    # results = run(filedict,print_progress)
    # end_time = time()
    # run_time = (end_time - begin_time) / 60
    # print("Final Results is :",results,filedict,run_time)