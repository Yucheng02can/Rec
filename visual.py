# -*- coding: utf-8 -*-
# @Time    : 2024/8/21 14:39
# @Author  : colagold
# @FileName: visual.py

# get item distribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings('ignore')

def plot_log(df,file_name):
    item_dist = df[1].value_counts()
    # Reset to default style, or use a suitable style
    plt.style.use('default')
    # Set the background colour to white
    plt.figure(facecolor='white')
    # picture (e.g. of life in the city)
    plt.plot(item_dist.values, color='red', linewidth=2)
    # Setting the x-axis and y-axis labels
    plt.xlabel('Item (log)', fontsize=14, labelpad=10)
    plt.ylabel('Number of users (log)', fontsize=15, labelpad=10)
    # Setting the Scale Label Font Size
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # Setting the scale of the axes to a logarithmic scale
    plt.xscale('log')
    plt.yscale('log')
    # Adjust margins
    plt.tight_layout()
    # Save the image, making sure the background is white
    plt.savefig(f'img/{file_name}_long_tail_log.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_distribution(df,file_name):
    item_dist = df[1].value_counts()
    # Reset to default style, or use a suitable style
    plt.style.use('default')
    # plot item distribution
    plt.figure()
    plt.plot(item_dist.values, color='red')
    plt.xlabel('Item', fontsize='14')
    plt.xticks(fontsize='10')
    plt.yticks(fontsize='13')
    plt.ylabel('Number of users', fontsize='15')
    plt.savefig(f'img/{file_name}_long_tail_normal.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_split(df, file_name):
    item_dist = df[1].value_counts()
    sorted_item_dist = item_dist.sort_values(ascending=False)
    cumulative_users = np.cumsum(sorted_item_dist.values)
    total_users = cumulative_users[-1]

    # Finding the threshold of 20 per cent of users
    top_20_percent_index = np.searchsorted(cumulative_users, 0.2 * total_users)

    # plot item distribution
    plt.style.use('default')
    plt.figure()
    plt.plot(sorted_item_dist.values, color='red')

    # Add 20 per cent dotted line
    plt.axvline(x=top_20_percent_index, color='blue', linestyle='--', label=f'Top 20% items')

    # Adding text labels
    plt.text(top_20_percent_index, max(sorted_item_dist.values) * 0.8, 'Top 20%',
             color='blue', fontsize=12, verticalalignment='center', horizontalalignment='right')

    plt.xlabel('Item', fontsize='14')
    plt.xticks(fontsize='10')
    plt.yticks(fontsize='13')
    plt.ylabel('Number of users', fontsize='15')
    plt.legend()
    plt.savefig(f'img/distribution/{file_name}_long_tail_normal.png', dpi=300, bbox_inches='tight')
    plt.show()


file_list=['ml-100k','ml-1m','lastfm','Book-Crossing']

for file_name in file_list:
    print(file_name)
    file_path=f"dataset/{file_name}.csv"
    df = pd.read_csv(file_path, header=None)
    plot_log(df,file_name)
    plot_distribution(df,file_name)
    plot_split(df,file_name)
