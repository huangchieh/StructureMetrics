#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity



if __name__ == '__main__':
    baseOut = 'output'
    structures = ['Prediction_1', 'Prediction_2', 'Prediction_3']

    # Common parameters for the plots, and output folder
    r_max = 3.5
    outputFolder = os.path.join(baseOut)
    print('Group 1')
    # --- RDF 
    # O-O 
    data = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, 'P')) 
    data1 = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, structures[0]))
    data2 = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, structures[1]))
    data3 = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, structures[2]))
    r, gr_OO = data['r'], data['gr']
    r_1, gr_OO_1 = data1['r'], data1['gr']
    r_2, gr_OO_2 = data2['r'], data2['gr']
    r_3, gr_OO_3 = data3['r'], data3['gr']
    print('Plotting RDF_OO')
    ylim= 3
    plot_rdf(r, [gr_OO, gr_OO_1, gr_OO_2, gr_OO_3 ], label='RDF_OO_123',  legend=['Reference', 'OO (Prediction 1)', 'OO (Prediction 2)', 'OO (Prediction 3)'], color=['#299035', '#fc0006', 'orange', 'black'], x_lim=r_max, y_lim=ylim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'], loc="upper left")
    s1 = cosine_similarity(gr_OO, gr_OO_1)
    s2 = cosine_similarity(gr_OO, gr_OO_2)
    s3 = cosine_similarity(gr_OO, gr_OO_3)
    print('Similarity between Reference and Prediction 1, 2, 3: ', s1, s2, s3)

    # O-H 
    data = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, 'P'))
    data1 = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, structures[0]))
    data2 = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, structures[1]))
    data3 = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, structures[2]))
    r, gr_OH = data['r'], data['gr']
    r_1, gr_OH_1 = data1['r'], data1['gr']
    r_2, gr_OH_2 = data2['r'], data2['gr']
    r_3, gr_OH_3 = data3['r'], data3['gr']
    print('Plotting RDF_OH')
    ylim= 20 
    plot_rdf(r, [gr_OH, gr_OH_1, gr_OH_2, gr_OH_3 ], label='RDF_OH_123',  legend=['Reference', 'OH (Prediction 1)', 'OH (Prediction 2)', 'OH (Prediction 3)'], color=['#299035', '#fc0006', 'orange', 'black'], x_lim=r_max, y_lim=ylim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'], loc="upper right")
    s1 = cosine_similarity(gr_OH, gr_OH_1)
    s2 = cosine_similarity(gr_OH, gr_OH_2)
    s3 = cosine_similarity(gr_OH, gr_OH_3)
    print('Similarity between Reference and Prediction 1, 2, 3: ', s1, s2, s3)

    # --- ADF
    # H-O-H 
    data = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'P', 'P'))
    data1 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Prediction_1', 'Prediction_1'))
    data2 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Prediction_2', 'Prediction_2'))
    data3 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Prediction_3', 'Prediction_3'))
    angles = data['angles']
    angles1 = data1['angles']
    angles2 = data2['angles']
    angles3 = data3['angles']
    print('Plotting ADF_HOH') 
    y_lim = 0.025
    bins = 120
    ns = plot_angle_distribution([angles, angles1, angles2, angles3], label='HOH_dist_123', legend=['Reference', 'HOH (Prediction 1)', 'HOH (Prediction 2)', 'HOH (Prediction 3)'], color=['#299035', '#fc0006', 'orange', 'black'], bins=bins, y_lim=y_lim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'])
    s1 = cosine_similarity(ns[0], ns[1])
    s2 = cosine_similarity(ns[0], ns[2])
    s3 = cosine_similarity(ns[0], ns[3])
    print('Similarity between Reference and Prediction 1, 2, 3: ', s1, s2, s3)

    # O-H-O 
    data = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'P', 'P'))
    data1 = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'Prediction_1', 'Prediction_1'))
    data2 = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'Prediction_2', 'Prediction_2'))
    data3 = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'Prediction_3', 'Prediction_3'))
    angles = data['angles']
    angles1 = data1['angles']
    angles2 = data2['angles']
    angles3 = data3['angles']
    print('Plotting ADF_OHO') 
    y_lim = 0.025
    bins = 120
    ns = plot_angle_distribution([angles, angles1, angles2, angles3], label='OHO_dist_123', legend=['Reference', 'OHO (Prediction 1)', 'OHO (Prediction 2)', 'OHO (Prediction 3)'], color=['#299035', '#fc0006', 'orange', 'black'], bins=bins, y_lim=y_lim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'], loc='upper right')
    s1 = cosine_similarity(ns[0], ns[1])
    s2 = cosine_similarity(ns[0], ns[2])
    s3 = cosine_similarity(ns[0], ns[3])
    print('Similarity between Reference and Prediction 1, 2, 3: ', s1, s2, s3)


    # Theta O-H 
    data = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'P', 'P'))
    data1 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Prediction_1', 'Prediction_1'))
    data2 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Prediction_2', 'Prediction_2'))
    data3 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Prediction_3', 'Prediction_3'))
    angles = data['angles']
    angles1 = data1['angles']
    angles2 = data2['angles']
    angles3 = data3['angles']
    print('Plotting ADF_ThetaOH') 
    y_lim = 0.035
    bins = 30 
    ns = plot_angle_distribution([angles, angles1, angles2, angles3], label='ThetaOH_dist_123', legend=['Reference', 'OH (Prediction 1)', 'OH (Prediction 2)', 'OH (Prediction 3)'], color=['#299035', '#fc0006', 'orange', 'black'], bins=bins, y_lim=y_lim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'], loc='upper right')
    s1 = cosine_similarity(ns[0], ns[1])
    s2 = cosine_similarity(ns[0], ns[2])
    s3 = cosine_similarity(ns[0], ns[3])
    print('Similarity between Reference and Prediction 1, 2, 3: ', s1, s2, s3)

    # For group 2
    print('Group 2')
    structures = ['Prediction_a', 'Prediction_b', 'Prediction_c']
    # Common parameters for the plots, and output folder
    r_max = 3.5
    outputFolder = os.path.join(baseOut)
    # --- RDF 
    # O-O 
    data = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, 'P')) 
    data1 = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, structures[0]))
    data2 = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, structures[1]))
    data3 = np.load('{}/{}/RDF_OO.npz'.format(outputFolder, structures[2]))
    r, gr_OO = data['r'], data['gr']
    r_1, gr_OO_1 = data1['r'], data1['gr']
    r_2, gr_OO_2 = data2['r'], data2['gr']
    r_3, gr_OO_3 = data3['r'], data3['gr']
    print('Plotting RDF_OO')
    ylim= 3
    plot_rdf(r, [gr_OO, gr_OO_1, gr_OO_2, gr_OO_3 ], label='RDF_OO_abc',  legend=['Reference', 'OO (Prediction a)', 'OO (Prediction b)', 'OO (Prediction c)'], color=['#299035', '#fc0006', 'orange', 'black'], x_lim=r_max, y_lim=ylim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'], loc="upper left")
    s1 = cosine_similarity(gr_OO, gr_OO_1)
    s2 = cosine_similarity(gr_OO, gr_OO_2)
    s3 = cosine_similarity(gr_OO, gr_OO_3)
    print('Similarity between Reference and Prediction a, b, c: ', s1, s2, s3)

    # O-H 
    data = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, 'P'))
    data1 = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, structures[0]))
    data2 = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, structures[1]))
    data3 = np.load('{}/{}/RDF_OH.npz'.format(outputFolder, structures[2]))
    r, gr_OH = data['r'], data['gr']
    r_1, gr_OH_1 = data1['r'], data1['gr']
    r_2, gr_OH_2 = data2['r'], data2['gr']
    r_3, gr_OH_3 = data3['r'], data3['gr']
    print('Plotting RDF_OH')
    ylim= 20 
    plot_rdf(r, [gr_OH, gr_OH_1, gr_OH_2, gr_OH_3 ], label='RDF_OH_abc',  legend=['Reference', 'OH (Prediction a)', 'OH (Prediction b)', 'OH (Prediction c)'], color=['#299035', '#fc0006', 'orange', 'black'], x_lim=r_max, y_lim=ylim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'], loc="upper right")
    s1 = cosine_similarity(gr_OH, gr_OH_1)
    s2 = cosine_similarity(gr_OH, gr_OH_2)
    s3 = cosine_similarity(gr_OH, gr_OH_3)
    print('Similarity between Reference and Prediction a, b, c: ', s1, s2, s3)

    # --- ADF
    # H-O-H 
    data = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'P', 'P'))
    data1 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Prediction_a', 'Prediction_a'))
    data2 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Prediction_b', 'Prediction_b'))
    data3 = np.load('{}/{}/HOH_dist_{}.npz'.format(outputFolder, 'Prediction_c', 'Prediction_c'))
    angles = data['angles']
    angles1 = data1['angles']
    angles2 = data2['angles']
    angles3 = data3['angles']
    print('Plotting ADF_HOH') 
    y_lim = 0.025
    bins = 120
    ns = plot_angle_distribution([angles, angles1, angles2, angles3], label='HOH_dist_abc', legend=['Reference', 'HOH (Prediction a)', 'HOH (Prediction b)', 'HOH (Prediction c)'], color=['#299035', '#fc0006', 'orange', 'black'], bins=bins, y_lim=y_lim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'])
    s1 = cosine_similarity(ns[0], ns[1])
    s2 = cosine_similarity(ns[0], ns[2])
    s3 = cosine_similarity(ns[0], ns[3])
    print('Similarity between Reference and Prediction a, b, c: ', s1, s2, s3)

    # O-H-O 
    data = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'P', 'P'))
    data1 = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'Prediction_a', 'Prediction_a'))
    data2 = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'Prediction_b', 'Prediction_b'))
    data3 = np.load('{}/{}/OHO_dist_{}.npz'.format(outputFolder, 'Prediction_c', 'Prediction_c'))
    angles = data['angles']
    angles1 = data1['angles']
    angles2 = data2['angles']
    angles3 = data3['angles']
    print('Plotting ADF_OHO') 
    y_lim = 0.025
    bins = 120
    ns = plot_angle_distribution([angles, angles1, angles2, angles3], label='OHO_dist_abc', legend=['Reference', 'OHO (Prediction a)', 'OHO (Prediction b)', 'OHO (Prediction c)'], color=['#299035', '#fc0006', 'orange', 'black'], bins=bins, y_lim=y_lim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'], loc='upper right')
    s1 = cosine_similarity(ns[0], ns[1])
    s2 = cosine_similarity(ns[0], ns[2])
    s3 = cosine_similarity(ns[0], ns[3])
    print('Similarity between Reference and Prediction a, b, c: ', s1, s2, s3)

    # Theta O-H 
    data = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'P', 'P'))
    data1 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Prediction_a', 'Prediction_a'))
    data2 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Prediction_b', 'Prediction_b'))
    data3 = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(outputFolder, 'Prediction_c', 'Prediction_c'))
    angles = data['angles']
    angles1 = data1['angles']
    angles2 = data2['angles']
    angles3 = data3['angles']
    print('Plotting ADF_ThetaOH') 
    y_lim = 0.035
    bins = 30
    ns = plot_angle_distribution([angles, angles1, angles2, angles3], label='ThetaHO_dist_abc', legend=['Reference', 'OH (Prediction a)', 'OH (Prediction b)', 'OH (Prediction c)'], color=['#299035', '#fc0006', 'orange', 'black'], bins=bins, y_lim=y_lim, outfolder=outputFolder, style=['bar', 'step', 'step', 'step'], loc='upper right')
    s1 = cosine_similarity(ns[0], ns[1])
    s2 = cosine_similarity(ns[0], ns[2])
    s3 = cosine_similarity(ns[0], ns[3])
    print('Similarity between Reference and Prediction a, b, c: ', s1, s2, s3)

