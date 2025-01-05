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
        # X_hda, Y_hda, Z_hda = plot_hbond_distance_vs_angle(hbonds, angle_type='hda', label='HBond_hda_{}'.format(structure), cmap='Greens')
        #return X_dha, Y_dha, Z_dha, X_hda, Y_hda, Z_hda
        return X_dha, Y_dha, Z_dha

if __name__ == '__main__':
    baseOut = 'images'
    results_file = os.path.join(baseOut, 'results_Hbond.json')

    _, _, Z_dha_p = calculate('Label')
    X_dha, Y_dha, Z_dha = calculate('Ref')
    Z_dha_diff_3 = Z_dha - Z_dha_p # This this reference case, results from the orginal PPAFM
    mse_dha_3 = np.mean(Z_dha_diff_3**2)

    # Structures predected by the different models
    structures = []
    structures_temp = ["PPAFM2Exp_CoAll_L{}_L{}_Elatest".format(L1, L2) for L1 in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] for L2 in [0.1, 1, 10]]
    structures.extend(structures_temp)    

    results = {}
    for structure in structures:
        X_dha, Y_dha, Z_dha = calculate(structure)
        Z_dha_diff_c = Z_dha - Z_dha_p  # Differences for Prediction_c
        mse_dha_c = np.mean(Z_dha_diff_c**2)

        relative_improvement_dha = (mse_dha_3 - mse_dha_c) / mse_dha_3
        absolute_improvement_dha = mse_dha_3 - mse_dha_c
        rmse_dha_3 = np.sqrt(mse_dha_3)
        rmse_dha_c = np.sqrt(mse_dha_c)
        mae_dha_c = np.mean(np.abs(Z_dha_diff_c))

        # Print the results
        print('Mean square error between Prediction_3 and Label: dha {:.5f}'.format(mse_dha_3))
        print('Mean square error between Prediction_c and Label: dha {:.5f}'.format(mse_dha_c))
        print('Relative improvement for Prediction_c compared to Prediction_3: dha {:.2f}'.format(relative_improvement_dha))
        print('Absolute improvement for Prediction_c compared to Prediction_3: dha {:.5f}'.format(absolute_improvement_dha))
        print('Root mean square error for Prediction_3: dha {:.5f}'.format(rmse_dha_3))
        print('Root mean square error for Prediction_c: dha {:.5f}'.format(rmse_dha_c))
        print('Mean absolute error for Prediction_c: dha {:.5f}'.format(mae_dha_c))

        # Store the results in the dictionary
        results[structure] = {
            'mse_dha_3': mse_dha_3,
            'mse_dha_c': mse_dha_c,
            'relative_improvement_dha': relative_improvement_dha,
            'absolute_improvement_dha': absolute_improvement_dha,
            'rmse_dha_3': rmse_dha_3,
            'rmse_dha_c': rmse_dha_c,
            'mae_dha_c': mae_dha_c,
        }

        outfolder = os.path.join(baseOut, structure)
        plot_density_difference(X_dha, Y_dha, Z_dha_diff_3, angle_type='dha', label='HBond_dha_diff_3', cmap='BrBG', outfolder=outfolder)
        plot_density_difference(X_dha, Y_dha, Z_dha_diff_c, angle_type='dha', label='HBond_dha_diff_c', cmap='BrBG', outfolder=outfolder)

    # Write the results to a JSON file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)