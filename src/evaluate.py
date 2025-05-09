#!/usr/bin/env python
import numpy as np
import pandas as pd
import json, os
import matplotlib.pyplot as plt

simcolor = '#ed9d2c'
expcolor = '#de461c'
bg07color = '#479FB1'
bv17color = '#6E7CBC'

plt.rcParams['font.size']=14
plt.rcParams['font.family']='Arial'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX

def plot_comparison_subplots(df0, df1, df2, numeric_columns, label0="Dataset 0", label1="Dataset 1", label2="Dataset 2", save_as=None, y_label=None):
    """
    Plots a comparison of mean values with error bars for three datasets.

    Parameters:
    df0 (pd.DataFrame): First dataset.
    df1 (pd.DataFrame): Second dataset.
    df2 (pd.DataFrame): Third dataset.
    numeric_columns (list): List of all numeric columns.
    label0 (str): Label for the first dataset.
    label1 (str): Label for the second dataset.
    label2 (str): Label for the third dataset.
    save_as (str, optional): File path to save the figure.

    Returns:
    None
    """
    global layer, show
    x_labels = {
        "OO": r"$d_{\mathrm{OO}}$", 
        "OH": r"$d_{\mathrm{OH}}$", 
        "HOH": r"$\theta_{\mathrm{HOH}}$", 
        "ZOH": r"$\theta_{\mathrm{ZOH}}$", 
        "Hbond": r"$(d_{\mathrm{O_d}\mathrm{O_a}}, \theta_{\mathrm{O_d}\mathrm{H}\mathrm{O_a}})$", 
        "OrderP": r"$(S_g, S_k)$"
    }
    x_ticklabels = [x_labels[col] for col in numeric_columns]
    # Compute mean and standard error for all datasets
    mean_values0 = df0[numeric_columns].mean()
    std_values0 = df0[numeric_columns].std() / np.sqrt(df0[numeric_columns].count())
    mean_values1 = df1[numeric_columns].mean()
    std_values1 = df1[numeric_columns].std() / np.sqrt(df1[numeric_columns].count())
    mean_values2 = df2[numeric_columns].mean()
    std_values2 = df2[numeric_columns].std() / np.sqrt(df2[numeric_columns].count())

    x = np.arange(len(numeric_columns))  # X positions for all features
    bar_width = 0.3  # Adjusted width for three bars

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot bars for all features
    ax.bar(x - bar_width, mean_values0, yerr=std_values0, capsize=5,
           alpha=0.7, width=bar_width, label=label0, color='lightgray', edgecolor='black')
    ax.bar(x, mean_values1, yerr=std_values1, capsize=5,
           alpha=0.7, width=bar_width, label=label1, color='skyblue', edgecolor='black')
    ax.bar(x + bar_width, mean_values2, yerr=std_values2, capsize=5,
           alpha=0.7, width=bar_width, label=label2, color='salmon', edgecolor='black')

    # Labels and titles
    ax.set_ylabel(y_label)
    ax.set_xticks(ticks=x)
    ax.set_xticklabels(x_ticklabels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    #ax.set_ylim(0, 0.35)
    ax.legend(loc='upper left')

    # Remove extra spaces between bars and axes
    ax.margins(x=0)

    # Adjust layout and show plot
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(f'{save_as}.png', dpi=600)
        plt.savefig(f'{save_as}.pdf')
        plt.savefig(f'{save_as}.svg')
    if show:
        plt.show()


def plotBarchat(model, df, df_P, df_ref, y_label):
    df_model = df[df["Structure"].str.contains("_{}_".format(model))]  # From selected model
    plot_comparison_subplots(df_P, df_ref, df_model, numeric_columns,
                             label0=rf"Labeled configurations ${{\bar{{\mathcal{{M}}}}}}$",
                             label1=rf"Predicted configurations ${{\tilde{{\mathcal{{M}}}}}}={{\mathcal{{F}}}}_{{\mathcal{{U}}}}({{\mathcal{{V}}}})$", 
                             label2=rf"Predicted configurations ${{\hat{{\mathcal{{M}}}}}}={{\mathcal{{F}}}}_{{\tilde{{\mathcal{{V}}}}}}({{\mathcal{{V}}}})$",
                             save_as='{}/{}_comparision_to_{}_{}_{}'.format(outputFolder, model, ground_truth, layer, y_label.replace(' ', '_')), 
                             y_label=y_label)

if __name__ == '__main__':
    show = True
    inputFolder = '../processed_data/distribution_distances'
    ground_truth, layer = 'Label', 'Top'
    #for layer in ['Top', 'All']:
    for layer in ['Top']:
        outputFolder = '../results/distance_evaluate'
        os.makedirs(outputFolder, exist_ok=True)
        with open('{}/similarities_{}_{}.json'.format(inputFolder, ground_truth, layer), "r") as file:
            similarities = json.load(file)

        numeric_columns = ["OO", "OH", "HOH",  "ZOH", "Hbond", "OrderP"]  # Changed order of ZOH and Hbond
        for distance, y_label in zip(['wdistancec_nor', 'edistancec_nor', 'mdistancec_nor'], ['Wasserstein distance', 'Energy distance', 'Maximum mean discrepancy ']):
            print('Distance:', distance)
            df = pd.DataFrame(columns = ['Structure', 'Truth', 'OO', 'OH', 'HOH', 'ZOH', 'Hbond', 'OrderP'])
            for i, (key, value) in enumerate(similarities.items()):
                df.loc[i] = [key, ground_truth, value['OO_dist'][distance], value['OH_dist'][distance], value['HOH_dist'][distance], value['ThetaOH_dist'][distance], value['Hbonds'][distance], value['OrderP'][distance]]
            df_ref = df[df["Structure"].str.contains("Ref")]
            df_P = df[df["Structure"] == 'P'] # The cropped version of Ref, including very large configurations
            print('L10_L10')
            plotBarchat('L10_L10', df, df_P, df_ref, y_label)
            print('L20_L1')
            plotBarchat('L20_L1', df, df_P, df_ref, y_label)
