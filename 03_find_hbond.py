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
        X_dha, Y_dha, Z_dha = plot_hbond_distance_vs_angle(hbonds, angle_type='dha', label='HBond_dha_{}'.format(structure), cmap='Greens')
        X_hda, Y_hda, Z_hda = plot_hbond_distance_vs_angle(hbonds, angle_type='hda', label='HBond_hda_{}'.format(structure), cmap='Greens')
        
        # Plot the density difference
        if structure == 'P':
            Z_dha_p, Z_hda_p = Z_dha, Z_hda
        if structure == 'Prediction_3':
            Z_dha_diff_3, Z_hda_diff_3 = Z_dha - Z_dha_p, Z_hda - Z_hda_p
        if structure == 'Prediction_c':
            Z_dha_diff_c, Z_hda_diff_c = Z_dha - Z_dha_p, Z_hda - Z_hda_p
    
    # Calculate the mean square error for dha and hda
    mse_dha_3 = np.mean(Z_dha_diff_3**2)
    mse_hda_3 = np.mean(Z_hda_diff_3**2)
    mse_dha_c = np.mean(Z_dha_diff_c**2)
    mse_hda_c = np.mean(Z_hda_diff_c**2)

    # Calculate the percentage improvement
    percent_improvement_dha = ((mse_dha_3 - mse_dha_c) / mse_dha_3) * 100
    percent_improvement_hda = ((mse_hda_3 - mse_hda_c) / mse_hda_3) * 100

    # Print the results
    print('Mean square error between Prediction_3 and Label: dha {:.5f}, hda {:.5f}'.format(mse_dha_3, mse_hda_3))
    print('Mean square error between Prediction_c and Label: dha {:.5f}, hda {:.5f}'.format(mse_dha_c, mse_hda_c))
    print('Percentage improvement for Prediction_c compared to Prediction_3: dha {:.2f}%, hda {:.2f}%'.format(percent_improvement_dha, percent_improvement_hda))

    plot_density_difference(X_dha, Y_dha, Z_dha_diff_3, angle_type='dha', label='HBond_dha_diff_3', cmap='BrBG')
    plot_density_difference(X_hda, Y_hda, Z_dha_diff_3, angle_type='hda', label='HBond_hda_diff_3', cmap='BrBG')
    plot_density_difference(X_dha, Y_dha, Z_dha_diff_c, angle_type='dha', label='HBond_dha_diff_c', cmap='BrBG')
    plot_density_difference(X_hda, Y_hda, Z_hda_diff_c, angle_type='hda', label='HBond_hda_diff_c', cmap='BrBG')
