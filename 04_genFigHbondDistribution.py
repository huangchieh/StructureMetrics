#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder, read_xyz_with_atomic_numbers
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution
from water import cal_all_hydrogen_bonds, plot_hbond_distance_vs_angle, plot_density_difference 

import json

def calculate(structure, outfolder, zThreshold=4.85, aboveZthres=None):
        # Read the samples
        # structure = structures[0]
        sampleFolder = os.path.join('BatchOutStructures', structure) if (structure == 'Label' or structure == 'P') else os.path.join('BatchOutStructures', structure, 'Prediction_c')
        samples = read_samples_from_folder(sampleFolder)
        print('Calculating for structure: {}'.format(structure))

        # Find hydrogen bonds
        print('Finding hydrogen bonds ...')
        #hbonds = cal_all_hydrogen_bonds(samples, z_min=9.5) # For the top layer water
        hbonds = cal_all_hydrogen_bonds(samples, zThresholdO=zThreshold, aboveZthres=aboveZthres) 
        print('Number of hydrogen bonds found: {}'.format(len(hbonds)))
        X_dha, Y_dha, Z_dha = plot_hbond_distance_vs_angle(hbonds, angle_type='dha', label='HBond_dha_{}_above_{}'.format(structure, str(aboveZthres)), cmap='Greens', figsize=(3, 2.2), outfolder=outfolder)
        return X_dha, Y_dha, Z_dha

if __name__ == '__main__':
    baseOut = 'Figures'
    X_dha, Y_dha, Z_dha = calculate('Label', outfolder=baseOut, aboveZthres=None)
    X_dha, Y_dha, Z_dha = calculate('Label', outfolder=baseOut, aboveZthres=True) 
    X_dha, Y_dha, Z_dha = calculate('Label', outfolder=baseOut, aboveZthres=False)

# Number of hydrogen bonds found for None, True, False: 11836, 1941, 9895