#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution

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
    results_file = os.path.join(imagePath, 'similarities.json')
    structures = [
        'PPAFM2Exp_CoAll_L60_L0_Elatest',
        'PPAFM2Exp_CoAll_L60_L0.1_Elatest',
        'PPAFM2Exp_CoAll_L60_L1_Elatest',
        'PPAFM2Exp_CoAll_L60_L10_Elatest',
        'PPAFM2Exp_CoAll_L50_L1_Elatest',
        'PPAFM2Exp_CoAll_L40_L1_Elatest',
        'PPAFM2Exp_CoAll_L20_L1_Elatest',
        'PPAFM2Exp_CoAll_L10_L1_Elatest',
    ]

    all_similarities = {}

    for structure in structures:
        print('Calculating for structure: {}'.format(structure))
        similarity_file = os.path.join(imagePath, structure, 'similarity.json')
        similarities = {}

        # Common parameters for the plots, and output folder
        r_max = 3.5
        outputFolder = os.path.join(baseOut)

        # --- RDF 
        # O-O 
        data = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, 'P')) 
        r, gr_OO = data['r'], data['gr']
        data3 = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, 'Ref'))
        r_3, gr_OO_3 = data3['r'], data3['gr']
        datac = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, structure))
        r_c, gr_OO_c = datac['r'], datac['gr']
        print('Plotting RDF_OO')
        ylim= 3
        if not os.path.exists('{}/{}'.format(imagePath, structure)):
            os.makedirs('{}/{}'.format(imagePath, structure))
        plot_rdf(r, [gr_OO, gr_OO_3, gr_OO_c], label='RDF_OO',  legend=['Reference', 'O-O (v0)', 'O-O (v1)'], color=['#299035', '#fc0006', '#215ab1'], x_lim=r_max, y_lim=ylim, outfolder='{}/{}'.format(imagePath, structure), style=['bar', 'step', 'step'], loc="upper left")
        s3 = cosine_similarity(gr_OO, gr_OO_3)
        sc = cosine_similarity(gr_OO, gr_OO_c)
        similarity_increase = ((sc - s3) / s3) * 100
        print('Similarity between Reference and Predictions: ', s3, sc)
        similarities['RDF_OO'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}

        # O-H 
        data = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, 'P'))
        r, gr_OH = data['r'], data['gr']
        data3 = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, "Ref"))
        r_3, gr_OH_3 = data3['r'], data3['gr']
        datac = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, structure))
        r_c, gr_OH_c = datac['r'], datac['gr']
        print('Plotting RDF_OH')
        ylim= 20 
        if not os.path.exists('{}/{}'.format(imagePath, structure)):
            os.makedirs('{}/{}'.format(imagePath, structure))
        plot_rdf(r, [gr_OH, gr_OH_3, gr_OH_c], label='RDF_OH',  legend=['Reference', 'O-H (v0)', 'O-H (v1)'], color=['#299035', '#fc0006', '#215ab1'], x_lim=r_max, y_lim=ylim, outfolder='{}/{}'.format(imagePath, structure), style=['bar', 'step', 'step'], loc="upper right")
        s3 = cosine_similarity(gr_OH, gr_OH_3)
        sc = cosine_similarity(gr_OH, gr_OH_c)
        similarity_increase = ((sc - s3) / s3) * 100
        print('Similarity between Reference and Predictions: ', s3, sc)
        similarities['RDF_OH'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}

        # --- ADF
        # H-O-H 
        data = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'P', 'P'))
        angles = data['angles']
        data3 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Ref', 'Ref'))
        angles3 = data3['angles']
        datac = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, structure, structure))
        anglesc = datac['angles'] 
        print('Plotting ADF_HOH') 
        y_lim = 0.025
        bins = 120
        if not os.path.exists('{}/{}'.format(imagePath, structure)):
            os.makedirs('{}/{}'.format(imagePath, structure))
        ns = plot_angle_distribution([angles, angles3, anglesc], label='HOH_dist', legend=['Reference', 'H-O-H (v0)', 'H-O-H (v1)'], color=['#299035', '#fc0006', '#215ab1'], bins=bins, y_lim=y_lim, outfolder='{}/{}'.format(imagePath, structure), style=['bar',  'step', 'step'])
        s3 = cosine_similarity(ns[0], ns[1])
        sc = cosine_similarity(ns[0], ns[2])
        similarity_increase = ((sc - s3) / s3) * 100
        print('Similarity between Reference and Predictions: ', s3, sc)
        similarities['ADF_HOH'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}

        # O-H-O 
        data = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'P', 'P'))
        angles = data['angles']
        data3 = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'Ref', 'Ref'))
        angles3 = data3['angles']
        datac = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, structure, structure))
        anglesc = datac['angles']
        print('Plotting ADF_OHO') 
        y_lim = 0.025
        bins = 120
        if not os.path.exists('{}/{}'.format(imagePath, structure)):
            os.makedirs('{}/{}'.format(imagePath, structure))
        ns = plot_angle_distribution([angles, angles3, anglesc], label='OHO_dist', legend=['Reference', 'O-H-O (v0)', 'O-H-O (v1)'], color=['#299035', '#fc0006', '#215ab1'], bins=bins, y_lim=y_lim, outfolder='{}/{}'.format(imagePath, structure), style=['bar', 'step', 'step'], loc='upper right')
        s3 = cosine_similarity(ns[0], ns[1])
        sc = cosine_similarity(ns[0], ns[2])
        similarity_increase = ((sc - s3) / s3) * 100
        print('Similarity between Reference and Predictions: ', s3, sc)
        similarities['ADF_OHO'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}

        # Theta O-H 
        data = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'P', 'P'))
        angles = data['angles']
        data3 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Ref', 'Ref'))
        angles3 = data3['angles']
        datac = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, structure, structure))
        anglesc = datac['angles']
        print('Plotting ADF_ThetaOH') 
        y_lim = 0.035
        bins = 30 
        if not os.path.exists('{}/{}'.format(imagePath, structure)):
            os.makedirs('{}/{}'.format(imagePath, structure))
        ns = plot_angle_distribution([angles, angles3, anglesc], label='ThetaOH_dist', legend=['Reference', 'O-H (v0)', 'O-H (v1)'], color=['#299035', '#fc0006', '#215ab1'], bins=bins, y_lim=y_lim, outfolder='{}/{}'.format(imagePath, structure), style=['bar', 'step', 'step'], loc='upper right')
        s3 = cosine_similarity(ns[0], ns[1])
        sc = cosine_similarity(ns[0], ns[2])
        similarity_increase = ((sc - s3) / s3) * 100
        print('Similarity between Reference and Predictions: ', s3, sc)
        similarities['ADF_ThetaOH'] = {'s3': s3, 'sc': sc, 'similarity_increase': similarity_increase}

        # Store similarities for the current structure
        all_similarities[structure] = similarities

    # Write all similarities to a single JSON file
    write_similarity_to_file(results_file, all_similarities)

