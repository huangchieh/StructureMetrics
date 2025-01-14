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
    baseOut = 'output'
    imagePath = 'images'
    ground_truth = 'Label'
    results_file = os.path.join(imagePath, 'similarities_{}.json'.format(ground_truth))

    # Structures predected by the different models
    structures = []
    structures_temp = ["PPAFM2Exp_CoAll_L{}_L{}_Elatest".format(L1, L2) for L1 in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] for L2 in [0.1, 1, 10]]
    #structures_temp = ["PPAFM2Exp_CoAll_L{}_L{}_Elatest".format(L1, L2) for L1 in [10] for L2 in [0.1]]
    structures.extend(structures_temp)

    # Record scores in all_similarities
    all_similarities = {}
    for structure in structures:
        print('Calculating for structure: {}'.format(structure))
        similarity_file = os.path.join(imagePath, structure, 'similarity.json')
        similarities = {}

        # Common parameters for the plots, and output folder
        outputFolder = os.path.join(baseOut)

        fig, axs = plt.subplots(4, 1, figsize=(6, 6))
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
        r_max = 3.5
        bins = 120
        plot_kde_fill(ax=axs[0], xmin=0, xmax=r_max, data=distances, color=colors['All'], linestyle=linestypes['All'], label='Ground truth', fill=fills['All'], alpha_fill=0.3)
        plot_kde_fill(ax=axs[0], xmin=0, xmax=r_max, data=distances3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=True)
        plot_kde_fill(ax=axs[0], xmin=0, xmax=r_max, data=distancesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=True)
        axs[0].set_xlabel(r'$r_\text{OO}$ (Å)')
        axs[0].set_ylabel(r'$\rho(r)$')
        axs[0].set_xlim(0, r_max)
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
        plot_kde_fill(ax=axs[1], xmin=0, xmax=r_max, data=distances3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=True)
        plot_kde_fill(ax=axs[1], xmin=0, xmax=r_max, data=distancesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=True)
        axs[1].set_xlabel(r'$r_\text{OH}$ (Å)')
        axs[1].set_ylabel(r'$\rho(r)$')
        axs[1].set_xlim(0, r_max)
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
        plot_kde_fill(ax=axs[2], xmin=0, xmax=180, data=angles3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=True)
        plot_kde_fill(ax=axs[2], xmin=0, xmax=180, data=anglesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=True)
        axs[2].set_xlabel(r'$\angle$HOH (degrees)')
        axs[2].set_xlim(0, 180)
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
        plot_kde_fill(ax=axs[3], xmin=0, xmax=180, data=angles3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=True)
        plot_kde_fill(ax=axs[3], xmin=0, xmax=180, data=anglesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=True)
        axs[3].set_xlabel(r'$\angle$ZOH (degrees)')
        axs[3].set_xlim(0, 180)
        axs[3].set_ylabel(r'$\rho(\theta)$')
        axs[3].tick_params(axis='both', direction='in')

        # Store similarities for the current structure
        all_similarities[structure] = similarities

        fig.subplots_adjust(hspace=0.4, wspace=0, left=0.1, bottom=0.1, right=0.95, top=0.98)
        plt.savefig(os.path.join(imagePath, structure, 'DistanceAngelDists.png'), dpi=300)
        #plt.show()
        plt.close()

        # Hbond
        data = np.load('{}/{}/Hbonds.npz'.format(outputFolder, ground_truth))['distance_angle']
        data3 = np.load('{}/{}/Hbonds.npz'.format(outputFolder, 'Ref'))['distance_angle']
        datac = np.load('{}/{}/Hbonds.npz'.format(outputFolder, structure))['distance_angle']
        wdistance3  = sinkhorn_2d_distance(data, data3)
        wdistancec  = sinkhorn_2d_distance(data, datac)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['Hbonds'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        plt.figure(figsize=(22, 6))
        # Plot PDF
        x_range_3, y_range_3, pdf, pdf_3  = kde(data, data3)
        x_range_c, y_range_c, pdf, pdf_c  = kde(data, datac)
        plt.subplot(1, 3, 1)
        plt.tick_params(axis='both', direction='in')
        contourf = plt.contourf(x_range_3, y_range_3, pdf, levels=30, cmap='Blues')
        subset = data[np.random.choice(data.shape[0], size=int(0.1 * data.shape[0]), replace=False)]
        plt.scatter(subset[:, 0], subset[:, 1], s=0.5, color='black', alpha=0.1)
        plt.xlim(0.8, 3.8)
        plt.ylim(120, 180)
        #plt.title("Data1 KDE PDF")
        plt.xlabel(r"$d_\text{OO}$ (Å)")
        plt.ylabel(r"$\angle$DHA (degrees)")
        plt.colorbar(contourf, label="Density")

        # Plot PDF3
        plt.subplot(1, 3, 2)
        plt.tick_params(axis='both', direction='in')
        contourf_3 = plt.contourf(x_range_3, y_range_3, pdf_3, levels=30, cmap='Reds')
        plt.scatter(data3[:, 0], data3[:, 1], s=0.5, color='black', alpha=0.1)
        #plt.title("Data2 KDE PDF")
        plt.xlim(0.8, 3.8)
        plt.ylim(120, 180)
        plt.xlabel(r"$d_\text{OO}$ (Å)")
        plt.ylabel(r"$\angle$DHA (degrees)")
        plt.colorbar(contourf_3, label="Density")

        # Plot PDFc
        plt.subplot(1, 3, 3)
        plt.tick_params(axis='both', direction='in')
        contourf_c = plt.contourf(x_range_c, y_range_c, pdf_c, levels=30, cmap='Greens')
        plt.scatter(datac[:, 0], datac[:, 1], s=0.5, color='black', alpha=0.1)
        #plt.title("Data2 KDE PDF")
        plt.xlim(0.8, 3.8)
        plt.ylim(120, 180)
        plt.xlabel(r"$d_\text{OO}$ (Å)")
        plt.ylabel(r"$\angle$DHA (degrees)")
        plt.colorbar(contourf_c, label="Density")

        plt.tight_layout()
        plt.savefig(os.path.join(imagePath, structure, 'Hbonds_dists.png'), dpi=300)
        #plt.show()
        plt.close()

    # Write all similarities to a single JSON file
    write_similarity_to_file(results_file, all_similarities)

