#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder, read_xyz_with_atomic_numbers
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution
from water import cal_all_hydrogen_bonds, plot_hbond_distance_vs_angle, plot_density_difference 

import json

def calculate(structure):
        # Read the samples
        # structure = structures[0]
        sampleFolder = os.path.join('BatchOutStructures', structure) if (structure == 'Label' or structure == 'P') else os.path.join('BatchOutStructures', structure, 'Prediction_c')
        samples = read_samples_from_folder(sampleFolder)
        print('Calculating for structure: {}'.format(structure))


        # Find hydrogen bonds
        print('Finding hydrogen bonds ...')
        #hbonds = cal_all_hydrogen_bonds(samples, z_min=9.5) # For the top layer water
        hbonds = cal_all_hydrogen_bonds(samples)
        print('Number of hydrogen bonds found: {}'.format(len(hbonds)))
        X_dha, Y_dha, Z_dha = plot_hbond_distance_vs_angle(hbonds, angle_type='dha', label='HBond_dha_{}'.format(structure), cmap='Greens')
        X_hda, Y_hda, Z_hda = plot_hbond_distance_vs_angle(hbonds, angle_type='hda', label='HBond_hda_{}'.format(structure), cmap='Greens')
    
        return X_dha, Y_dha, Z_dha, X_hda, Y_hda, Z_hda

if __name__ == '__main__':
    baseOut = 'images'
    results_file = os.path.join(baseOut, 'results_Hbond.json')

    _, _, Z_dha_p, _, _, Z_hda_p = calculate('P')
    X_dha, Y_dha, Z_dha, X_hda, Y_hda, Z_hda = calculate('Ref')
    Z_dha_diff_3, Z_hda_diff_3 = Z_dha - Z_dha_p, Z_hda - Z_hda_p # This this reference case, results from the orginal PPAFM
    mse_dha_3 = np.mean(Z_dha_diff_3**2)
    mse_hda_3 = np.mean(Z_hda_diff_3**2)

    structures = [ 'PPAFM2Exp_CoAll_L60_L0_Elatest', \
                  'PPAFM2Exp_CoAll_L60_L0.1_Elatest', \
                  'PPAFM2Exp_CoAll_L60_L1_Elatest', \
                  'PPAFM2Exp_CoAll_L60_L10_Elatest', \
                  'PPAFM2Exp_CoAll_L50_L1_Elatest', \
                  'PPAFM2Exp_CoAll_L40_L1_Elatest', \
                  'PPAFM2Exp_CoAll_L20_L1_Elatest', \
                  'PPAFM2Exp_CoAll_L10_L1_Elatest', \
                  ]
    results = {}
    for structure in structures:
        X_dha, Y_dha, Z_dha, X_hda, Y_hda, Z_hda = calculate(structure)
        Z_dha_diff_c, Z_hda_diff_c = Z_dha - Z_dha_p, Z_hda - Z_hda_p # This is the results from the Fake AFM
        mse_dha_c = np.mean(Z_dha_diff_c**2)
        mse_hda_c = np.mean(Z_hda_diff_c**2)

        percent_improvement_dha = ((mse_dha_3 - mse_dha_c) / mse_dha_3) * 100
        percent_improvement_hda = ((mse_hda_3 - mse_hda_c) / mse_hda_3) * 100

        # Print the results
        print('Mean square error between Prediction_3 and Label: dha {:.5f}, hda {:.5f}'.format(mse_dha_3, mse_hda_3))
        print('Mean square error between Prediction_c and Label: dha {:.5f}, hda {:.5f}'.format(mse_dha_c, mse_hda_c))
        print('Percentage improvement for Prediction_c compared to Prediction_3: dha {:.2f}%, hda {:.2f}%'.format(percent_improvement_dha, percent_improvement_hda))

        # Store the results in the dictionary
        results[structure] = {
            'mse_dha_3': mse_dha_3,
            'mse_hda_3': mse_hda_3,
            'mse_dha_c': mse_dha_c,
            'mse_hda_c': mse_hda_c,
            'percent_improvement_dha': percent_improvement_dha,
            'percent_improvement_hda': percent_improvement_hda
        }

        outfolder = os.path.join(baseOut, structure)
        plot_density_difference(X_dha, Y_dha, Z_dha_diff_3, angle_type='dha', label='HBond_dha_diff_3', cmap='BrBG', outfolder=outfolder)
        plot_density_difference(X_hda, Y_hda, Z_dha_diff_3, angle_type='hda', label='HBond_hda_diff_3', cmap='BrBG', outfolder=outfolder)
        plot_density_difference(X_dha, Y_dha, Z_dha_diff_c, angle_type='dha', label='HBond_dha_diff_c', cmap='BrBG', outfolder=outfolder)
        plot_density_difference(X_hda, Y_hda, Z_hda_diff_c, angle_type='hda', label='HBond_hda_diff_c', cmap='BrBG', outfolder=outfolder)

    # Write the results to a JSON file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)