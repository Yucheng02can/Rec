# -*- coding: utf-8 -*-
# @Time    : 2024/8/21 14:24
# @FileName: dataset_statistics.py
import os

import pandas as pd


def calculate_sparsity(data):
    num_users = data[0].nunique()
    num_items = data[1].nunique()
    num_interactions = len(data)
    sparsity = 1 - (num_interactions / (num_users * num_items))
    return sparsity


def print_describe(df):
    print(df.describe())
    print(f"Number of users.{df[0].nunique()}")
    print(f"Number of items.{df[1].nunique()}")
    print(f"{file_name} Sparsity of.{calculate_sparsity(df)}")

file_name='ml-100k'
file_list=['ml-100k','ml-1m','lastfm','Book-Crossing']

for file_name in file_list:
    print(file_name)
    file_path=f"dataset/{file_name}.csv"
    df = pd.read_csv(file_path, header=None)
    print_describe(df)