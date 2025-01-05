#!/usr/bin/env python

# 1. Load data from the output folder and plot RDF and ADF for the reference and predicted structures
# 2. Calculate the cosine similarity between the reference and predicted RDF and ADF

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from water import plot_rdf, plot_angle_distribution, plot_kde_fill  
from water import compute_kde_wasserstein, compute_kde_wasserstein_2d
import json

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
    structures.extend(structures_temp)

    # Record scores in all_similarities
    all_similarities = {}
    for structure in structures:
        print('Calculating for structure: {}'.format(structure))
        similarity_file = os.path.join(imagePath, structure, 'similarity.json')
        similarities = {}

        # Common parameters for the plots, and output folder
        #r_max = 3.5
        outputFolder = os.path.join(baseOut)

        fig, axs = plt.subplots(4, 1, figsize=(6, 6))
        axs = axs.flatten()
        colors = {'All': 'green', 'V0': 'red', 'V1': 'black'}
        linestypes = {'All': '-', 'V0': ':', 'V1': '-.'}
        fills = {'All': True, 'V0': True, 'V1': False}
        # --- RDF 
        # O-O 
        # data = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, ground_truth)) 
        # r, gr_OO = data['r'], data['gr']
        # data3 = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, 'Ref'))
        # r_3, gr_OO_3 = data3['r'], data3['gr']
        # datac = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, structure))
        # r_c, gr_OO_c = datac['r'], datac['gr']
        # print('Plotting RDF_OO')
        # ylim= 3
        # if not os.path.exists('{}/{}'.format(imagePath, structure)):
        #     os.makedirs('{}/{}'.format(imagePath, structure))
        # plot_rdf(r, [gr_OO, gr_OO_3, gr_OO_c], label='RDF_OO',  legend=['Reference', 'O-O (v0)', 'O-O (v1)'], color=['#299035', '#fc0006', '#215ab1'], x_lim=r_max, y_lim=ylim, outfolder='{}/{}'.format(imagePath, structure), style=['bar', 'step', 'step'], loc="upper left")
        # s3 = cosine_similarity(gr_OO, gr_OO_3)
        # sc = cosine_similarity(gr_OO, gr_OO_c)
        # similarity_increase = ((sc - s3) / s3) * 100
        # print('Similarity between Reference and Predictions: ', s3, sc)
        # similarities['RDF_OO'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}

        # O-H 
        # data = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, ground_truth))
        # r, gr_OH = data['r'], data['gr']
        # data3 = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, "Ref"))
        # r_3, gr_OH_3 = data3['r'], data3['gr']
        # datac = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, structure))
        # r_c, gr_OH_c = datac['r'], datac['gr']
        # print('Plotting RDF_OH')
        # ylim= 20 
        # if not os.path.exists('{}/{}'.format(imagePath, structure)):
        #     os.makedirs('{}/{}'.format(imagePath, structure))
        # plot_rdf(r, [gr_OH, gr_OH_3, gr_OH_c], label='RDF_OH',  legend=['Reference', 'O-H (v0)', 'O-H (v1)'], color=['#299035', '#fc0006', '#215ab1'], x_lim=r_max, y_lim=ylim, outfolder='{}/{}'.format(imagePath, structure), style=['bar', 'step', 'step'], loc="upper right")
        # s3 = cosine_similarity(gr_OH, gr_OH_3)
        # sc = cosine_similarity(gr_OH, gr_OH_c)
        # similarity_increase = ((sc - s3) / s3) * 100
        # print('Similarity between Reference and Predictions: ', s3, sc)
        # similarities['RDF_OH'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}

        # --- Distance distribution
        # O-O
        data = np.load('{}/{}/OO_distances.npz'.format(outputFolder, ground_truth))
        distances = data['distances']
        data3 = np.load('{}/{}/OO_distances.npz'.format(outputFolder, 'Ref'))
        distances3 = data3['distances']
        datac = np.load('{}/{}/OO_distances.npz'.format(outputFolder, structure))
        distancesc = datac['distances']
        wdistance3 = compute_kde_wasserstein(distances, distances3) 
        wdistancec = compute_kde_wasserstein(distances, distancesc)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['OO_dist'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

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
        wdistance3 = compute_kde_wasserstein(distances, distances3)
        wdistancec = compute_kde_wasserstein(distances, distancesc)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['OH_dist'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        r_max = 1.25 
        plot_kde_fill(ax=axs[1], xmin=0, xmax=r_max, data=distances, color=colors['All'], linestyle=linestypes['All'], label='Ground truth', fill=fills['All'], alpha_fill=0.3)
        plot_kde_fill(ax=axs[1], xmin=0, xmax=r_max, data=distances3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=True)
        plot_kde_fill(ax=axs[1], xmin=0, xmax=r_max, data=distancesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=True)
        axs[1].set_xlabel(r'$r_\text{OH}$ (Å)')
        axs[1].set_ylabel(r'$\rho(r)$')
        axs[1].set_xlim(0, r_max)
        #axs[1].set_xlim(0.9, r_max-0.15)
        axs[1].tick_params(axis='both', direction='in')

        # --- ADF
        # H-O-H 
        data = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, ground_truth, ground_truth))
        angles = data['angles']
        data3 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Ref', 'Ref'))
        angles3 = data3['angles']
        datac = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, structure, structure))
        anglesc = datac['angles'] 
        # print('Plotting ADF_HOH') 
        # y_lim = 0.025
        # bins = 120
        # if not os.path.exists('{}/{}'.format(imagePath, structure)):
        #     os.makedirs('{}/{}'.format(imagePath, structure))
        # ns = plot_angle_distribution([angles, angles3, anglesc], label='HOH_dist', legend=['Reference', 'H-O-H (v0)', 'H-O-H (v1)'], color=['#299035', '#fc0006', '#215ab1'], bins=bins, y_lim=y_lim, outfolder='{}/{}'.format(imagePath, structure), style=['bar',  'step', 'step'])
        # s3 = cosine_similarity(ns[0], ns[1])
        # sc = cosine_similarity(ns[0], ns[2])
        # similarity_increase = ((sc - s3) / s3) * 100
        # print('Similarity between Reference and Predictions: ', s3, sc)
        # similarities['ADF_HOH'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}

        wdistance3 = compute_kde_wasserstein(angles, angles3)
        wdistancec = compute_kde_wasserstein(angles, anglesc)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['HOH_dist'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

        plot_kde_fill(ax=axs[2], xmin=0, xmax=180, data=angles, color=colors['All'], linestyle=linestypes['All'], label='Ground truth', fill=fills['All'], alpha_fill=0.3)
        plot_kde_fill(ax=axs[2], xmin=0, xmax=180, data=angles3, color=colors['V0'], linestyle=linestypes['V0'], label='V0', fill=fills['V0'], alpha_fill=0.3, hist=True)
        plot_kde_fill(ax=axs[2], xmin=0, xmax=180, data=anglesc, color=colors['V1'], linestyle=linestypes['V1'], label='V1', fill=fills['V1'], alpha_fill=0.3, hist=True)
        axs[2].set_xlabel(r'$\angle$HOH (degrees)')
        axs[2].set_xlim(0, 180)
        axs[2].set_ylabel(r'$\rho(\theta)$')
        axs[2].tick_params(axis='both', direction='in')

        # # O-H-O 
        # data = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, ground_truth, ground_truth))
        # angles = data['angles']
        # data3 = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'Ref', 'Ref'))
        # angles3 = data3['angles']
        # datac = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, structure, structure))
        # anglesc = datac['angles']
        # # print('Plotting ADF_OHO') 
        # # y_lim = 0.025
        # # bins = 120
        # # if not os.path.exists('{}/{}'.format(imagePath, structure)):
        # #     os.makedirs('{}/{}'.format(imagePath, structure))
        # # ns = plot_angle_distribution([angles, angles3, anglesc], label='OHO_dist', legend=['Reference', 'O-H-O (v0)', 'O-H-O (v1)'], color=['#299035', '#fc0006', '#215ab1'], bins=bins, y_lim=y_lim, outfolder='{}/{}'.format(imagePath, structure), style=['bar', 'step', 'step'], loc='upper right')
        # # s3 = cosine_similarity(ns[0], ns[1])
        # # sc = cosine_similarity(ns[0], ns[2])
        # # similarity_increase = ((sc - s3) / s3) * 100
        # # print('Similarity between Reference and Predictions: ', s3, sc)
        # # similarities['ADF_OHO'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}
        # wdistance3 = compute_kde_wasserstein(angles, angles3)
        # wdistancec = compute_kde_wasserstein(angles, anglesc)
        # wdistance_decrease = wdistance3 - wdistancec
        # similarities['OHO_dist'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}


        # Theta O-H 
        data = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, ground_truth, ground_truth))
        angles = data['angles']
        data3 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Ref', 'Ref'))
        angles3 = data3['angles']
        datac = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, structure, structure))
        anglesc = datac['angles']
        # print('Plotting ADF_ThetaOH') 
        # y_lim = 0.035
        # bins = 30 
        # if not os.path.exists('{}/{}'.format(imagePath, structure)):
        #     os.makedirs('{}/{}'.format(imagePath, structure))
        # ns = plot_angle_distribution([angles, angles3, anglesc], label='ThetaOH_dist', legend=['Reference', 'O-H (v0)', 'O-H (v1)'], color=['#299035', '#fc0006', '#215ab1'], bins=bins, y_lim=y_lim, outfolder='{}/{}'.format(imagePath, structure), style=['bar', 'step', 'step'], loc='upper right')
        # s3 = cosine_similarity(ns[0], ns[1])
        # sc = cosine_similarity(ns[0], ns[2])
        # similarity_increase = ((sc - s3) / s3) * 100
        # print('Similarity between Reference and Predictions: ', s3, sc)
        # similarities['ADF_ThetaOH'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}
        wdistance3 = compute_kde_wasserstein(angles, angles3)
        wdistancec = compute_kde_wasserstein(angles, anglesc)
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
        wdistance3, x_range, y_range, pdf1, pdf2  = compute_kde_wasserstein_2d(data, data3)
        wdistancec, x_range, y_range, pdf1, pdf2  = compute_kde_wasserstein_2d(data, datac)
        wdistance_decrease = wdistance3 - wdistancec
        similarities['Hbonds'] = {'wdistance3': wdistance3, 'wdistancec': wdistancec, 'wdistance_decrease': wdistance_decrease}

    # Write all similarities to a single JSON file
    write_similarity_to_file(results_file, all_similarities)

