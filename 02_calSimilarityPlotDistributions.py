#!/usr/bin/env python

# 1. Load data from the output folder and plot RDF and ADF for the reference and predicted structures
# 2. Calculate the distance between the reference and predicted RDF and ADF

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from water import plot_rdf, plot_angle_distribution, plot_kde_fill  
from water import sinkhorn_2d_distance, kde
import json
from scipy.stats import wasserstein_distance
import argparse 

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity

# Function to write similarities to file
def write_similarity_to_file(file_path, similarities):
    with open(file_path, 'w') as f:
        json.dump(similarities, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testStability", action="store_true", help="Use to test the stability")
    args = parser.parse_args()
    print("Test stability:", args.testStability)
    baseOut = 'output'
    imagePath = 'images'
    ground_truth = 'Label'
    results_file = os.path.join(imagePath, 'similarities_{}{}.json'.format('testStability_' if args.testStability else '', ground_truth))

    # Structures predected by the different models
    structures = []
    if args.testStability:
        structures_temp = ["PPAFM2Exp_CoAll_L10_L10_Elatest_C{}".format(c) for c in range(0, 10)]
    else:
        structures_temp = ["PPAFM2Exp_CoAll_L{}_L{}_Elatest".format(L1, L2) for L1 in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] for L2 in [0.1, 1, 10]]
    #structures_temp = ["PPAFM2Exp_CoAll_L{}_L{}_Elatest".format(L1, L2) for L1 in [10] for L2 in [0.1]]
    structures.extend(structures_temp)

    # Record scores in all_similarities
    all_similarities = {}
    for structure in structures:
        print('Calculating for structure: {}'.format(structure))
        #os.path.join(imagePath, structure)
        os.makedirs(os.path.join(imagePath, structure), exist_ok=True)
        #similarity_file = os.path.join(imagePath, structure, 'similarity.json' if not argparse.args.testStability else 'similarity_testStability.json')
        similarities = {}

        # Common parameters for the plots, and output folder
        outputFolder = os.path.join(baseOut)

        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs = axs.flatten()
        colors = {'All': 'green', 'V0': 'red', 'V1': 'black'}
        linestypes = {'All': '-', 'V0': ':', 'V1': '-.'}
        fills = {'All': True, 'V0': True, 'V1': False}

        # --- Distance distribution
        # O-O
        # Record distances and their changes
        data = np.load('{}/{}/OO_distances.npz'.format(outputFolder, ground_truth))
        distances = data['distances']
        data3 = np.load('{}/{}/OO_distances.npz'.format(outputFolder, 'Ref'))
        distances3 = data3['distances']
        datac = np.load('{}/{}/OO_distances.npz'.format(outputFolder, structure))
        distancesc = datac['distances']
        wdistance3 = wasserstein_distance(distances, distances3) 
        wdistancec = wasserstein_distance(distances, distancesc)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['OO_dist'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        # Visulize 
        show_plots = False
        r_max = 3.5
        bins = 120
        hist = True
        plot_kde_fill(ax=axs[0], xmin=0, xmax=r_max, data=distances, color=colors['All'], linestyle=linestypes['All'], label='Reference', fill=fills['All'], alpha_fill=0.3)
        plot_kde_fill(ax=axs[0], xmin=0, xmax=r_max, data=distances3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=hist)
        plot_kde_fill(ax=axs[0], xmin=0, xmax=r_max, data=distancesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=hist)
        axs[0].set_xlabel(r'$r_\text{OO}$ (Å)')
        axs[0].set_ylabel(r'$\rho(r)$')
        axs[0].set_xlim(1.5, r_max)
        axs[0].set_ylim(bottom=0)
        axs[0].legend(frameon=False, ncol=1)
        axs[0].tick_params(axis='both', direction='in')

        # O-H
        data = np.load('{}/{}/OH_distances.npz'.format(outputFolder, ground_truth))
        distances = data['distances']
        data3 = np.load('{}/{}/OH_distances.npz'.format(outputFolder, 'Ref'))
        distances3 = data3['distances']
        datac = np.load('{}/{}/OH_distances.npz'.format(outputFolder, structure))
        distancesc = datac['distances']
        wdistance3 = wasserstein_distance(distances, distances3)
        wdistancec = wasserstein_distance(distances, distancesc)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['OH_dist'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        r_max = 1.25 
        plot_kde_fill(ax=axs[1], xmin=0, xmax=r_max, data=distances, color=colors['All'], linestyle=linestypes['All'], label='Ground truth', fill=fills['All'], alpha_fill=0.3)
        plot_kde_fill(ax=axs[1], xmin=0, xmax=r_max, data=distances3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=hist)
        plot_kde_fill(ax=axs[1], xmin=0, xmax=r_max, data=distancesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=hist)
        axs[1].set_xlabel(r'$r_\text{OH}$ (Å)')
        axs[1].set_ylabel(r'$\rho(r)$')
        axs[1].set_xlim(0.6, r_max)
        axs[1].set_ylim(bottom=0)
        axs[1].tick_params(axis='both', direction='in')

        # --- ADF
        # H-O-H 
        data = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, ground_truth, ground_truth))
        angles = data['angles']
        data3 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Ref', 'Ref'))
        angles3 = data3['angles']
        datac = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, structure, structure))
        anglesc = datac['angles'] 
        wdistance3 = wasserstein_distance(angles, angles3)
        wdistancec = wasserstein_distance(angles, anglesc)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['HOH_dist'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        plot_kde_fill(ax=axs[2], xmin=0, xmax=180, data=angles, color=colors['All'], linestyle=linestypes['All'], label='Ground truth', fill=fills['All'], alpha_fill=0.3)
        plot_kde_fill(ax=axs[2], xmin=0, xmax=180, data=angles3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=hist)
        plot_kde_fill(ax=axs[2], xmin=0, xmax=180, data=anglesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=hist)
        axs[2].set_xlabel(r'$\angle$HOH (degrees)')
        axs[2].set_xlim(50, 150)
        axs[2].set_ylim(bottom=0)
        axs[2].set_ylabel(r'$\rho(\theta)$')
        axs[2].tick_params(axis='both', direction='in')


        # Theta O-H 
        data = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, ground_truth, ground_truth))
        angles = data['angles']
        data3 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Ref', 'Ref'))
        angles3 = data3['angles']
        datac = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, structure, structure))
        anglesc = datac['angles']
        wdistance3 = wasserstein_distance(angles, angles3)
        wdistancec = wasserstein_distance(angles, anglesc)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['ThetaOH_dist'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        plot_kde_fill(ax=axs[3], xmin=0, xmax=180, data=angles, color=colors['All'], linestyle=linestypes['All'], label='Ground truth', fill=fills['All'], alpha_fill=0.3)
        plot_kde_fill(ax=axs[3], xmin=0, xmax=180, data=angles3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=hist, bins=40)
        plot_kde_fill(ax=axs[3], xmin=0, xmax=180, data=anglesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=hist, bins=40)
        axs[3].set_xlabel(r'$\angle$ZOH (degrees)')
        axs[3].set_xlim(0, 180)
        axs[3].set_ylim(bottom=0)
        axs[3].set_ylabel(r'$\rho(\theta)$')
        axs[3].tick_params(axis='both', direction='in')

        # Store similarities for the current structure
        all_similarities[structure] = similarities

        fig.subplots_adjust(hspace=0.2, wspace=0.3, left=0.1, bottom=0.1, right=0.95, top=0.98)
        plt.savefig(os.path.join(imagePath, structure, 'DistanceAngelDists.png'), dpi=600)
        if show_plots:
            plt.show()
        plt.close()

        # Hbond
        data = np.load('{}/{}/Hbonds.npz'.format(outputFolder, ground_truth))['distance_angle']
        data3 = np.load('{}/{}/Hbonds.npz'.format(outputFolder, 'Ref'))['distance_angle']
        datac = np.load('{}/{}/Hbonds.npz'.format(outputFolder, structure))['distance_angle']
        wdistance3  = sinkhorn_2d_distance(data, data3)
        wdistancec  = sinkhorn_2d_distance(data, datac)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['Hbonds'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        plt.figure(figsize=(10, 2.5))
        # Plot PDF
        x_range_3, y_range_3, pdf, pdf_3  = kde(data, data3)
        x_range_c, y_range_c, pdf, pdf_c  = kde(data, datac)
        x_min = np.min([x_range_3.min(), x_range_c.min()])
        x_max = np.max([x_range_3.max(), x_range_c.max()])
        y_min = np.min([y_range_3.min(), y_range_c.min()])
        y_max = np.max([y_range_3.max(), y_range_c.max()])
        def plot_sub(subNum, data, pdf, xlabel, ylabel, text, cmap, x_min=0.8, x_max=3.8, y_min=120, y_max=180, percent=0.1, logY=False, levels=30):
            plt.subplot(1, 3, subNum)
            plt.tick_params(axis='both', direction='in')
            if logY:
                plt.yscale('log') 
            #contourf = plt.contourf(x_range_3, y_range_3, pdf, levels=30, cmap=cmap)
            contour = plt.pcolormesh(x_range_3, y_range_3, pdf, cmap=cmap)
            plt.colorbar(contour, label="Density")
            # if text == 'Reference' and percent < 1.0:
            #     subset = data[np.random.choice(data.shape[0], size=int(percent * data.shape[0]), replace=False)]
            #     plt.scatter(subset[:, 0], subset[:, 1], s=0.5, color='black', alpha=0.1)
            # else:
            #     print('Plotting all data')
            #     plt.scatter(data[:, 0], data[:, 1], s=0.5, color='black', alpha=0.1)
            plt.scatter(data[:, 0], data[:, 1], s=0.5, color='black', alpha=0.1)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            # Add text to the top left corner of the plot
            plt.text(0.05, 0.95, text, color='black', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            #plt.colorbar(contourf, label="Density")

        xlabel = r"$d_\text{OO}$ (Å)"
        ylabel = r"$\angle$DHA (degrees)"
        plot_sub(1, data, pdf, xlabel, ylabel, 'Reference', 'Blues', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        plot_sub(2, data3, pdf_3, xlabel, ylabel, 'V0', 'Reds', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        plot_sub(3, datac, pdf_c, xlabel, ylabel, 'V1', 'Greens', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        plt.tight_layout()
        plt.savefig(os.path.join(imagePath, structure, 'Hbonds_dists.png'), dpi=600)
        if show_plots:
            plt.show()
        plt.close()

        #######################
        # Order parameter 2d
        #######################
        print('OrderP')
        data = np.load('{}/{}/OrderP.npz'.format(outputFolder, ground_truth))['sg_sk']
        data3 = np.load('{}/{}/OrderP.npz'.format(outputFolder, 'Ref'))['sg_sk']
        datac = np.load('{}/{}/OrderP.npz'.format(outputFolder, structure))['sg_sk']
        wdistance3  = sinkhorn_2d_distance(data, data3)
        wdistancec  = sinkhorn_2d_distance(data, datac)
        wdistance_decrease = wdistance3 - wdistancec
        print('OrderP, wdistance3: {:.5f}, wdistancec: {:.5f}, wdistance_decrease: {:.5f}'.format(wdistance3, wdistancec, wdistance_decrease))
        similarities['OrderP'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        plt.figure(figsize=(10, 2.5))
        # Plot PDF
        x_range_3, y_range_3, pdf, pdf_3  = kde(data, data3, grid_size=200)
        x_range_c, y_range_c, pdf, pdf_c  = kde(data, datac, grid_size=200)
        x_min = np.min([x_range_3.min(), x_range_c.min()])
        x_max = np.max([x_range_3.max(), x_range_c.max()])
        y_min = np.min([y_range_3.min(), y_range_c.min()])
        y_max = np.max([y_range_3.max(), y_range_c.max()])
        xlabel = r"$S_g$"
        ylabel = r"$S_k$"
        plot_sub(1, data, pdf, xlabel, ylabel, 'Reference', 'Blues', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, levels=60)
        plot_sub(2, data3, pdf_3, xlabel, ylabel, 'V0', 'Reds', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, levels=60)
        plot_sub(3, datac, pdf_c, xlabel, ylabel, 'V1', 'Greens', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, levels=60)

        plt.tight_layout()
        plt.savefig(os.path.join(imagePath, structure, 'OrderP_dists.png'), dpi=600)
        if show_plots:
            plt.show()
        plt.close()

        # # Order parameter
        # fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        # axs = axs.flatten() 

        # def one_subplot(ax, prop='d5', xlabel=r'$d_5$ (Å)', ylabel=r'$\rho(d_5)$', x_min=2.6, x_max=6.5):
        #     # d5
        #     data = np.load('{}/{}/{}.npz'.format(outputFolder, ground_truth, prop))
        #     prop_s = data[prop]
        #     data3 = np.load('{}/{}/{}.npz'.format(outputFolder, 'Ref', prop))
        #     prop_s3 = data3[prop]
        #     datac = np.load('{}/{}/{}.npz'.format(outputFolder, structure, prop))
        #     prop_sc = datac[prop]
        #     wdistance3 = wasserstein_distance(prop_s, prop_s3)
        #     wdistancec = wasserstein_distance(prop_s, prop_sc)
        #     wdistance_decrease = wdistance3 - wdistancec
        #     print('{}, wdistance3: {:.5f}, wdistancec: {:.5f}, wdistance_decrease: {:.5f}'.format(prop, wdistance3, wdistancec, wdistance_decrease))
        #     similarities['{}_dist'.format(prop)] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        #     plot_kde_fill(ax=ax, xmin=x_min, xmax=x_max, data=prop_s, color=colors['All'], linestyle=linestypes['All'], label='Ground truth', fill=fills['All'], alpha_fill=0.3)
        #     plot_kde_fill(ax=ax, xmin=x_min, xmax=x_max, data=prop_s3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=hist)
        #     plot_kde_fill(ax=ax, xmin=x_min, xmax=x_max, data=prop_sc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=hist)
        #     ax.set_xlabel(xlabel)
        #     ax.set_ylabel(ylabel)
        #     #axs[1].set_xlim(0.6, r_max)
        #     ax.set_ylim(bottom=0)
        #     ax.tick_params(axis='both', direction='in')
        #     # Legend
        #     ax.legend(frameon=False, ncol=1)
        # one_subplot(axs[0], prop='d5', xlabel=r'$d_5$ (Å)', ylabel=r'$\rho(d_5)$')
        # one_subplot(axs[1], prop='sg', xlabel=r'$q$', ylabel=r'$\rho(q)$', x_min=-2.3, x_max=1)
        # one_subplot(axs[2], prop='sk', xlabel=r'$S_k$', ylabel=r'$\rho(S_k)$', x_min=0.9, x_max=1)
        # one_subplot(axs[3], prop='lsi', xlabel=r'$LSI$ (Å)', ylabel=r'$\rho(\text{LSI})$', x_min=0, x_max=0.23)
        # if show_plots:
        #     plt.show()
        # plt.close()

    # Write all similarities to a single JSON file
    write_similarity_to_file(results_file, all_similarities)

