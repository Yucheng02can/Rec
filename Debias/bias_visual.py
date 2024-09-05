# -*- coding: utf-8 -*-
# @Time    : 2024/6/16 17:29
# @Author  : colagold
# @FileName: bias_and_coldstart_visual2.py

# Reads the entire Excel file and returns a dictionary where the key is the sheet name and the value is the corresponding DataFrame.
import copy
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings

from matplotlib.font_manager import FontProperties

font="Times New Roman"
warnings.filterwarnings('ignore')
excel_file_path = r'result\汇总.xlsx'
excel_data = pd.read_excel(excel_file_path, sheet_name=None)
dataname_dic={
    "BookCrossing":"BookCrossing",
    "lastfm":"Lastfm",
    "ml100k":"ml-100k",
    "ml1m":"ml-1m"
}

model_dic={
    "LightGCN":"LightGCN",
    "MF":"MF",
    "NeuMF":"NeuMF",
    "NGCF":"NGCF",
    "EnGCN":"ENGCN"
}

# Iterate through the dictionary to get the DataFrame for each sheet
def get_sheet(excel_data):
    data_dict={}

    for sheet_name, df in excel_data.items():
        # Access to experimental data
        if sheet_name in ["Amazon_Movies_and_TV10","dianping10","Amazon_CDs_and_Vinyl10","netflix10"]:
            continue
        df=df.dropna() # Remove the experimental data and delete the empty rows.
        data_dict[sheet_name]=df
        print(f"Sheet Name: {sheet_name}")
    return data_dict

def select_data(df_dicts,save_list,column,label="cold"):
    df_dict=copy.deepcopy(df_dicts)
    for data_name, df in df_dict.items():
        # Access to experimental data
        new_col_name=df[save_list].T.iloc[0, :].values
        new_df=df[save_list].T
        new_df.columns=new_col_name
        df=new_df[column].drop('model', axis=0) #Getting model data
        df=df[column].apply(pd.to_numeric, errors='coerce') #Convert to float
        df=df.round(4) # Retain valid numbers


        # Replace index name
        # Here we replace the indexes from 0, 1, 2 with ‘index 1’, ‘index 2’, ‘index 3’
        if label=="group":
            old_index=list(df.index)
            new_index=["0-15","15-20","20-25","25-30","30-35"]
        else:
            old_index = list(df.index)
            new_index = ["HEAD", "TAIL"]
        new_index_names = dict(zip(old_index,new_index))
        df = df.rename(index=new_index_names)
        df.rename(columns=model_dic, inplace=True)
        df_dict[data_name] = df  # Take the required experimental data
    return df_dict

def set_xy_fontsize(ax,font_size=21):
    ax.set_yticklabels(ax.get_yticklabels(), fontfamily=font, fontsize=font_size)
    # Set the font and size of the x-axis labels and align them horizontally
    plt.xticks(rotation=0, fontfamily=font, fontsize=font_size)

def set_legend(font_size=21):
    font_prop = FontProperties(family=font, size=font_size)
    plt.legend(prop=font_prop,  framealpha=0)  # ,framealpha=0


def plot_legend(df, data_name):
    plt.figure(figsize=(10, 5))
    copy_df=copy.deepcopy(df)
    colors = ["#8ECFC9", "#FFBE7A", "#BEB8DC", "#FA7F6F", "#E7DAD2"]
    copy_df.iloc[:, :] = np.nan
    ax = copy_df.plot(kind="bar", color=colors)
    # Adding a Twisted Dot Pattern
    hatch_patterns = ['/', 'o', '*', 'x', '\\', '+', '.', 'O']
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(hatch_patterns[-2])
    # Generate legends and arrange them horizontally
    #set_legend(font_size=15)

    plt.axis('off')
    # Generate legends and arrange them horizontally
    font_prop = FontProperties(family=font, size=15)
    plt.legend(ncol=5, prop=font_prop, loc='center left', framealpha=0)
    #set_legend(font_size=21)
    # Save Legend
    plt.savefig(f"../img/bias_visual/legend.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_bar(df,data_name):
    colors = ["#8ECFC9", "#FFBE7A", "#BEB8DC", "#E7DAD2", "#FA7F6F"]
    ax=df.plot(kind="bar",color=colors)

    # Adding a Twisted Dot Pattern
    hatch_patterns = ['/', 'o', '*', 'x', '\\', '+', '.', 'O']  # 可以使用各种不同的填充图案
    # Add pockmarks to each column
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(hatch_patterns[-2])

    # Set the interval of the y-axis
    max_value = df.max().max()
    plt.ylim(0, math.ceil(max_value * 100) / 100 + 0.005)

    # Setting up chart titles and labels
    # ax.set_title(‘Example Bar Chart’)
    # ax.set_xlabel(‘Model’)
    ax.set_ylabel('Recall@20',fontdict={'fontsize': 15, 'fontfamily': font})
    # Legend not shown
    #ax.legend_.remove()
    set_xy_fontsize(ax,font_size=21)
    plt.tight_layout()
    set_legend(font_size=15)
    #E:\Desktop\ Thesis Pictures
    plt.savefig(f"../img/bias_visual/{data_name}.png", dpi=300)
    plt.show()



if __name__=="__main__":
    # Data adaptation
    metric="RECALL20"
    cold_list=["model",f"0-15_{metric}",f"15-20_{metric}",f"20-25_{metric}",f"25-30_{metric}",f"30-35_{metric}"]
    baias_list=["model",f"Head_{metric}",f"Tail_{metric}"]
    model_list = ["MF", "NeuMF","NGCF","LightGCN","EnGCN"]
    sheet_dict = get_sheet(excel_data)
    cold_df_dict = select_data(sheet_dict, cold_list, model_list,label="group")
    for data_name,cold_data in cold_df_dict.items():
        print(data_name)
        print(cold_data)
        plot_bar(cold_data,metric+"Group_Bias_"+data_name)

    bias_df_dict = select_data(sheet_dict, baias_list, model_list,label="bias")
    for data_name, bias_data in bias_df_dict.items():
        print(data_name)
        print(bias_data)
        plot_bar(bias_data,metric+"Head_and_Tail_"+data_name)