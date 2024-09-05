import os
import time

import pandas as pd


def filter_best(path:str,metric:dict,args:list):
    csv_list = os.listdir(path)
    metric_dict=dict() #{metric_name:DataFrame}
    for metric,optimal in metric.items():

        metric_lines = None
        for csv_file in csv_list:
            if not csv_file.endswith(".csv"):
                continue
            if csv_file.startswith("result"):continue
            df = pd.read_csv(os.path.join(path, csv_file))


            if csv_file.split("-")[1].endswith(".csv"):
                alg, data_set_name = csv_file.split("-")[0], csv_file.split("-")[1].split(".")[0]
            else:
                alg, data_set_name = csv_file.split("-")[0], csv_file.split("-")[1]+csv_file.split("-")[2].split(".")[0]
            if data_set_name in ["Amazon_Movies_and_TV10","dianping10","Amazon_CDs_and_Vinyl10","netflix10"]:
                continue
            col_names = df.columns.tolist()
            col_names.insert(0, "data_set")
            col_names.insert(1, "model")

            if optimal=="min":
                print(alg)
                line = df[df[metric]==df[metric].min()]
            elif optimal=="max":
                line = df[df[metric]==df[metric].max()]
            else:
                print("Please make sure that it is treated as a maximum or minimum value.")
            line = line.reindex(columns=col_names)
            line["model"]=alg
            line["data_set"]=data_set_name
            metric_lines=pd.concat([metric_lines,line])







        metric_dict[metric]=metric_lines
        print(metric,optimal)
        print(metric_lines)
    return metric_dict

def delete_by_arg(df,arg_name:list):
    return df.drop(columns=arg_name)

def filter_by_dataset(path,metric_dict:dict,column_go_behind_list):

    for metric,df in metric_dict.items():

        excel_path = f"result/汇总.xlsx"
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        grouped=df.groupby('data_set')
        for name, group in grouped:

            group=group.drop_duplicates(subset=[metric])

            group = group.dropna(axis=1, how='any')


            del_line=["svdpp"]
            group = group[~group["model"].isin(del_line)]

            group.to_excel(writer, sheet_name=name, index=False)



        writer.save()


def combine_metric(label,metrics):
    m=[]
    for l in label:
        for metric in metrics:
            if l !="":
                m.append(l+"_"+metric)
            else:
                m.append(metric)
    return m





def column_go_behind(col_list:list,target_col:str):
    return [col for col in col_list if col not in target_col] + target_col

if __name__=="__main__":
    path = 'result/temp_result' #
    metrics={"NDCG20":"max"}
    delete_arg_list=["cache_dir"]
    column_go_behind_list=["tensorboard_path"]
    metrics_dict=filter_best(path,metrics,delete_arg_list)
    filter_by_dataset(path,metrics_dict,column_go_behind_list)