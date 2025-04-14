#!/usr/bin/env python

import numpy as np
import pandas as pd
import json, os
import matplotlib.pyplot as plt


plt.rcParams['font.size']=14

def plot_comparison_subplots(df0, df1, df2, numeric_columns, small_range_features, large_range_features, label0="Dataset 0", label1="Dataset 1", label2="Dataset 2", save_as=None):
    """
    Plots a comparison of mean values with error bars for three datasets,
    using two subplots: one for small-range features and one for large-range features.

    Parameters:
    df0 (pd.DataFrame): First dataset.
    df1 (pd.DataFrame): Second dataset.
    df2 (pd.DataFrame): Third dataset.
    numeric_columns (list): List of all numeric columns.
    small_range_features (list): Features with small values to be plotted separately.
    large_range_features (list): Features with larger values.
    label0 (str): Label for the first dataset.
    label1 (str): Label for the second dataset.
    label2 (str): Label for the third dataset.
    save_as (str, optional): File path to save the figure.

    Returns:
    None
    """
    global layer, show
    # Compute mean and standard error for all datasets
    mean_values0 = df0[numeric_columns].mean()
    std_values0 = df0[numeric_columns].std()/np.sqrt(df0[numeric_columns].count())
    mean_values1 = df1[numeric_columns].mean()
    std_values1 = df1[numeric_columns].std()/np.sqrt(df1[numeric_columns].count())
    mean_values2 = df2[numeric_columns].mean()
    std_values2 = df2[numeric_columns].std()/np.sqrt(df2[numeric_columns].count())

    # Define bar width and positions
    x1 = np.arange(len(small_range_features))  # X positions for small range features
    x2 = np.arange(len(large_range_features))  # X positions for large range features
    bar_width = 0.3  # Adjusted width for three bars

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(7, 5))

    # Plot for small range features
    axes[0].bar(x1 - bar_width, mean_values0[small_range_features], yerr=std_values0[small_range_features], capsize=5,
                alpha=0.7, width=bar_width, label=label0, color='lightgray', edgecolor='black')
    axes[0].bar(x1, mean_values1[small_range_features], yerr=std_values1[small_range_features], capsize=5,
                alpha=0.7, width=bar_width, label=label1, color='skyblue', edgecolor='black')
    axes[0].bar(x1 + bar_width, mean_values2[small_range_features], yerr=std_values2[small_range_features], capsize=5,
                alpha=0.7, width=bar_width, label=label2, color='salmon', edgecolor='black')

    # Plot for large range features
    axes[1].bar(x2 - bar_width, mean_values0[large_range_features], yerr=std_values0[large_range_features], capsize=5,
                alpha=0.7, width=bar_width, label=label0, color='lightgray', edgecolor='black')
    axes[1].bar(x2, mean_values1[large_range_features], yerr=std_values1[large_range_features], capsize=5,
                alpha=0.7, width=bar_width, label=label1, color='skyblue', edgecolor='black')
    axes[1].bar(x2 + bar_width, mean_values2[large_range_features], yerr=std_values2[large_range_features], capsize=5,
                alpha=0.7, width=bar_width, label=label2, color='salmon', edgecolor='black')

    # Labels and titles for both subplots
    axes[0].set_ylabel("Wasserstein Distance")
    axes[0].set_xticks(ticks=x1)
    axes[0].set_xticklabels(small_range_features, rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].text(x=0.05, y=0.65, s=f"Ref.: {layer} layer", transform=axes[0].transAxes)
    axes[0].set_ylim(0, 0.4)    
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].legend(loc='upper left')

    #axes[1].set_ylabel("Wasserstein Distance")
    axes[1].yaxis.tick_right()
    axes[1].set_xticks(ticks=x2)
    axes[1].set_xticklabels(large_range_features, rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 50)    
    axes[1].spines['left'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    #axes[1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    if show: plt.show()


def plotBarchat(model):
    df_model = df[df["Structure"].str.contains("_{}_".format(model))] # From selected model
    plot_comparison_subplots(df_P, df_ref, df_model, numeric_columns, ['OO', 'OH', 'OrderP'],
                             ['HOH', 'ZOH', 'Hbond'], label0="Training data",
                             label1="Baseline", label2=model,
                             save_as='{}/{}_comparision_to_{}_{}.png'.format(outputFolder, model, ground_truth, layer))


if __name__ == '__main__':
    show = False
    inputFolder = '../processed_data/distribution_distances'
    ground_truth, layer = 'Label', 'Top'
    for layer in ['Top', 'Bottom', 'All']:
        outputFolder = '../results/distance_evaluate'
        os.makedirs(outputFolder, exist_ok=True)
        # Load json file
        with open('{}/similarities_{}_{}.json'.format(inputFolder, ground_truth, layer), "r") as file:
            similarities = json.load(file)
        
        numeric_columns = ["OO", "OH", "HOH", "ZOH", "Hbond", "OrderP"]
        df = pd.DataFrame(columns = ['Structure', 'Truth', 'OO', 'OH', 'HOH', 'ZOH', 'Hbond', 'OrderP'])
        for i, (key, value) in enumerate(similarities.items()):
            df.loc[i] = [key, ground_truth, value['OO_dist']['wdistancec'], value['OH_dist']['wdistancec'], value['HOH_dist']['wdistancec'], value['ThetaOH_dist']['wdistancec'], value['Hbonds']['wdistancec'], value['OrderP']['wdistancec']]
        
        
        df_ref = df[df["Structure"].str.contains("Ref")]
        df_P = df[df["Structure"] == 'P'] # The cropped version of Ref, including very large configurations
        plotBarchat('L10_L10')
        plotBarchat('L20_L1')
        plotBarchat('L20_L10')
    
