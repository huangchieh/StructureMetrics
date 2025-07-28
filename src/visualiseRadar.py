#!/usr/bin/env python
import numpy as np
import pandas as pd
import json, os
import matplotlib.pyplot as plt
from utils import radar_plot

simcolor = '#ed9d2c'
expcolor = '#de461c'
bg07color = '#479FB1'
bv17color = '#6E7CBC'

plt.rcParams['font.size']=14
plt.rcParams['font.family']='Arial'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX


if __name__ == '__main__':
    show = False
    inputFolder = '../processed_data/distribution_distances'
    ground_truth, layer = 'Label', 'Top'
    for layer in ['Top', 'All']:
        print('Comparing layer:', layer)
        outputFolder = '../results/radar'
        os.makedirs(outputFolder, exist_ok=True)
        with open('{}/similarities_{}_{}.json'.format(inputFolder, ground_truth, layer), "r") as file:
            similarities = json.load(file)

        numeric_columns = ["OO", "OH", "HOH",  "ZOH", "Hbond", "OrderP"]  # Changed order of ZOH and Hbond
        # Get the key list 
        all_keys = list(similarities.keys())
        for i, comp_key in enumerate(all_keys):
            print('Comparing: ', comp_key)
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
            fig.suptitle('Comparison of {} layer: {}'.format(layer, comp_key))
            index = 0
            for distance, y_label in zip(['wdistancec_nor', 'edistancec_nor', 'mdistancec_nor'], ['Wasserstein distance', 'Energy distance', 'Maximum mean discrepancy']):
                print('Distance type:', distance)
                df = pd.DataFrame(columns = ['Structure', 'Truth', 'OO', 'OH', 'HOH', 'ZOH', 'Hbond', 'OrderP'])
                for i, (key, value) in enumerate(similarities.items()):
                    df.loc[i] = [key, ground_truth, value['OO_dist'][distance], value['OH_dist'][distance], value['HOH_dist'][distance], value['ThetaOH_dist'][distance], value['Hbonds'][distance], value['OrderP'][distance]]
                # Find the min and max values for each column for normalization
                min_values = df[numeric_columns].min()
                max_values = df[numeric_columns].max()
                min_values = np.array(min_values)
                max_values = np.array(max_values)
                print('Min values:', min_values)
                print('Max values:', max_values)

                # Find the reference performance
                ref_key = "Ref_Pure"
                ref_df = df[df["Structure"].str.contains(ref_key)]
                print(ref_df)
                # Get the numeric columns and compute the mean and std or ref_df 
                mean_values = ref_df[numeric_columns].mean()
                std_values = ref_df[numeric_columns].std() / np.sqrt(ref_df[numeric_columns].count())
                print('Mean', mean_values)
                print('Std', std_values)
                labels = [r"$d_{\mathrm{OO}}$", r"$d_{\mathrm{OH}}$", r"$\theta_{\mathrm{HOH}}$", r"$\theta_{\mathrm{ZOH}}$", r"$(d_{\mathrm{O_d}\mathrm{O_a}}, \theta_{\mathrm{O_d}\mathrm{H}\mathrm{O_a}})$", r"$(S_k, S_g)$"]
                title = y_label 
                radar_plot(ax=axs[index], mins=min_values, maxs=max_values, data=mean_values, color='k', errors=std_values, labels=labels, title=title)
                # Individual plots for each model
                compare_data = df[df["Structure"].str.contains(comp_key)] 
                compare_data = compare_data[numeric_columns].mean()
                radar_plot(ax=axs[index], mins=min_values, maxs=max_values, data=compare_data, color=simcolor, errors=None, labels=labels)
                index += 1
            plt.tight_layout()
            plt.savefig('{}/{}_comparision_to_{}_{}.png'.format(outputFolder, comp_key, ground_truth, layer))
            if show: plt.show() 