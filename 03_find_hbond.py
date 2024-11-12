#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder, read_xyz_with_atomic_numbers
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution
from water import cal_all_hydrogen_bonds, plot_hbond_distance_vs_angle, plot_density_difference 



if __name__ == '__main__':
    baseOut = 'output'
    #os.makedirs(outputFolder, exist_ok=True)
    # structures = ['Label', 'P', 'Prediction_1', 'Prediction_2', 'Prediction_3', \
    #               'Prediction_a', 'Prediction_b', 'Prediction_c']
    structures = ['P', 'Prediction_3', 'Prediction_c']
    #structures = ['P', 'Prediction_3', 'Prediction_c']

    for structure in structures:
        # Create the output folders
        os.makedirs(os.path.join(baseOut, structure), exist_ok=True)

        # Read the samples
        # structure = structures[0]
        samples = read_samples_from_folder(os.path.join('Structures', structure))
        print('Calculating for structure: {}'.format(structure))


        # Find hydrogen bonds
        print('Finding hydrogen bonds ...')
        #hbonds = cal_all_hydrogen_bonds(samples, z_min=9.5) # For the top layer water
        hbonds = cal_all_hydrogen_bonds(samples)
        X_dha, Y_dha, Z_dha = plot_hbond_distance_vs_angle(hbonds, angle_type='dha', label='HBond_dha_{}'.format(structure), cmap='viridis', use_density_estimate=True)
        X_hda, Y_hda, Z_hda = plot_hbond_distance_vs_angle(hbonds, angle_type='hda', label='HBond_hda_{}'.format(structure), cmap='viridis', use_density_estimate=True)
        
        # Plot the density difference
        if structure == 'P':
            Z_dha_p, Z_hda_p = Z_dha, Z_hda
        if structure == 'Prediction_3':
            Z_dha_diff_3, Z_hda_diff_3 = Z_dha - Z_dha_p, Z_hda - Z_hda_p
        if structure == 'Prediction_c':
            Z_dha_diff_c, Z_hda_diff_c = Z_dha - Z_dha_p, Z_hda - Z_hda_p
    
    print('Mean square error between Prediction_3 and Label: dha {:.5f}, hda {:.5f}'.format(np.mean(Z_dha_diff_3**2), np.mean(Z_hda_diff_3**2)))
    print('Mean square error between Prediction_c and Label: dha {:.5f}, hda {:.5f}'.format(np.mean(Z_dha_diff_c**2), np.mean(Z_hda_diff_c**2)))
    plot_density_difference(X_dha, Y_dha, Z_dha_diff_3, angle_type='dha', label='HBond_dha_diff_3', cmap='BrBG', use_density_estimate=True)
    plot_density_difference(X_hda, Y_hda, Z_hda_diff_3, angle_type='hda', label='HBond_hda_diff_3', cmap='BrBG', use_density_estimate=True)
    plot_density_difference(X_dha, Y_dha, Z_dha_diff_c, angle_type='dha', label='HBond_dha_diff_c', cmap='BrBG', use_density_estimate=True)
    plot_density_difference(X_hda, Y_hda, Z_hda_diff_c, angle_type='hda', label='HBond_hda_diff_c', cmap='BrBG', use_density_estimate=True)
