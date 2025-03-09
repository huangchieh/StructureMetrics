#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder, read_xyz_with_atomic_numbers
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution
from water import cal_all_hydrogen_bonds, plot_hbond_distance_vs_angle, plot_density_difference 

from scipy.stats import gaussian_kde

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
    # X_dha, Y_dha, Z_dha = calculate('Label', outfolder=baseOut, aboveZthres=None)
    # X_dha, Y_dha, Z_dha = calculate('Label', outfolder=baseOut, aboveZthres=True) 
    # X_dha, Y_dha, Z_dha = calculate('Label', outfolder=baseOut, aboveZthres=False)
    # Number of hydrogen bonds found for None, True, False: 11836, 1941, 9895

    structure = 'Label'
    sampleFolder = os.path.join('BatchOutStructures', structure) if (structure == 'Label' or structure == 'P') else os.path.join('BatchOutStructures', structure, 'Prediction_c')
    samples = read_samples_from_folder(sampleFolder)
    print('Calculating for structure: {}'.format(structure))

    outputFolder = os.path.join(baseOut, structure)
    z_thresholds = {'All': None, 'Top': True, 'Bottom': False}

    r_max = 3.5
    hbonds = cal_all_hydrogen_bonds(samples) 
    distances = np.array([hb[3] for hb in hbonds])
    angles = np.array([hb[4] for hb in hbonds])
    x_min, y_min = distances.min(), angles.min()
    x_max, y_max = 3.5, 180
    nbin = 50
    figsize = (8, 2.5)
    cmap = 'Greens'
    xgrid = np.linspace(x_min, x_max, nbin)
    ygrid = np.linspace(y_min, y_max, nbin)
    X, Y = np.meshgrid(xgrid, ygrid)
    texts = ['All', 'Top', 'Bottom']

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.025}, constrained_layout=True)
    for k, (key, value) in enumerate(z_thresholds.items()):
        hbonds = cal_all_hydrogen_bonds(samples, aboveZthres=value, zThresholdO=4.85) 
        distances = np.array([hb[3] for hb in hbonds])
        angles = np.array([hb[4] for hb in hbonds])
        xy = np.vstack([distances, angles])
        kde = gaussian_kde(xy)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)

        contour = axes[k].pcolormesh(X, Y, Z, cmap=cmap, vmin=0, vmax=0.22)
        axes[k].scatter(distances, angles, s=0.5, color='black', alpha=0.1)
        axes[k].tick_params(axis='both', direction='in', top=True, right=True)
        axes[k].text(0.2, 0.95, texts[k], color='black', fontsize=10, transform=axes[k].transAxes, verticalalignment='top')
        axes[k].set_xlim(x_min, x_max)
        axes[k].set_ylim(y_min, y_max)
        axes[k].set_xlabel('$d_{OO}$ (Å)')
        if k == 0:
            axes[k].set_ylabel(r'$\angle$DHA (°)')
    
    # Create shared colorbar
    cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label(r'$\rho(d, \theta)$')

    plt.savefig('{}/Hbond2D.png'.format(outputFolder), dpi=600)
    plt.savefig('{}/Hbond2D.svg'.format(outputFolder))
    plt.show()