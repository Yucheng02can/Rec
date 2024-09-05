import os.path
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def calculate_sparsity(data):
    num_users = data[0].nunique()
    num_items = data[1].nunique()
    num_interactions = len(data)
    sparsity = 1 - (num_interactions / (num_users * num_items))
    return sparsity

def print_describe(df):
    print(df.describe())
    print(f"Number of users:{df[0].nunique()}")
    print(f"Number of projects:{df[1].nunique()}")
    print(f"{file_name} sparsity:{calculate_sparsity(df)}")

#List construction method
def replace(df):
    # mapping table
    user_id = df[0].unique()
    user_dic = dict(zip(user_id, [i + 1 for i in range(user_id.shape[0])]))

    item_id = df[1].unique()
    item_dic = dict(zip(item_id, [i + 1 for i in range(item_id.shape[0])]))
    user_tmp,item_tmp=[],[]
    data=zip(df[0],df[1])
    for user,item in tqdm(data,total=df[0].size):
        user_tmp.append(user_dic[user])
        item_tmp.append(item_dic[item])
    df[0]=user_tmp
    df[1]=item_tmp
    return df

def sample_interactions(data, target_interactions):
    # Calculate sampling ratios
    sample_ratio = target_interactions / len(data)

    # Random sampling
    sampled_data = data.sample(frac=sample_ratio, random_state=42)

    return sampled_data

file_name='ml-100k'
file_list=['ml-100k','Amazon_Books','Yelp2018','Amazon_Beauty']
file_list=['Amazon_Books','Yelp2018','Amazon_Beauty']
file_list=['lastfm']

for file_name in file_list:
    print(file_name)
    path = 'naive_data\{}\{}.inter'.format(file_name,file_name)
    file_path=f"dataset/{file_name}.csv"
    if os.path.exists(file_path):
        print("File already exists")
        df = pd.read_csv(file_path, header=None)
        print_describe(df)
        continue
    if file_name=='lastfm':
        df = pd.read_table(path, header=None).iloc[1:][[0,1,2]]
    else:
        df=pd.read_table(path, header=None, delimiter='\\s',engine='python',encoding='utf-8')

    print("Read all data successfully")
    # The movielens have already been processed, just convert them to csv files!
    if file_name.startswith("ml-"):
        print("Handling movielens")
        df.iloc[1:].to_csv("dataset/{}.csv".format(file_name),header=False, index=False)
        print_describe(df)
        continue
    # Set a target number of interactions
    target_interactions = 100000

    # # Sampling
    # sampled_df = sample_interactions(df, target_interactions)
    #
    # # Print an overview of the data after sampling
    # print("Post-sampling data profile:")
    # print(sampled_df.describe())
    # df=sampled_df

    # Record start time
    start_time = datetime.now()
    file_path=f"dataset/{file_name}.csv"

    df=replace(df)
    print_describe(df)
    df.to_csv("dataset/{}.csv".format(file_name),header=False, index=False)
    time2 = datetime.now()
    print("Replacement successful, time consumed.",time2-start_time)




