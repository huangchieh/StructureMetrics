#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution


if __name__ == '__main__':
    baseOut = 'output'
    #os.makedirs(outputFolder, exist_ok=True)
    structures = ['Label', 'P', \
                  'Ref', \
                  'PPAFM2Exp_CoAll_L60_L0_Elatest', \
                  'PPAFM2Exp_CoAll_L60_L0.1_Elatest', \
                  'PPAFM2Exp_CoAll_L60_L1_Elatest', \
                  'PPAFM2Exp_CoAll_L60_L10_Elatest', \
                  'PPAFM2Exp_CoAll_L50_L1_Elatest', \
                  'PPAFM2Exp_CoAll_L40_L1_Elatest', \
                  'PPAFM2Exp_CoAll_L20_L1_Elatest', \
                  'PPAFM2Exp_CoAll_L10_L1_Elatest', \
                  ]

    for structure in structures:
        # Create the output folders
        os.makedirs(os.path.join(baseOut, structure), exist_ok=True)

        # Read the samples
        sampleFolder = os.path.join('BatchOutStructures', structure) if (structure == 'Label' or structure == 'P') else os.path.join('BatchOutStructures', structure, 'Prediction_c')
        samples = read_samples_from_folder(sampleFolder)
        print('Calculating for structure: {}'.format(structure))
        # Common parameters for the plots, and output folder
        r_max = 3.5
        #mic = True if structure == 'Label' else False
        mic = False
        bins = 120
        color = '#299035'
        outputFolder = os.path.join(baseOut, structure)
        # --- RDF 
        print('Calulating RDF ...')
        # O-O 
        r, gr_OO = mean_rdf(samples, 'O', 'O', r_max=r_max, mic=mic, bins=bins)
        label = 'RDF_OO_{}'.format(structure)
        legend = 'OO ({})'.format(structure) if structure  !=  'P' else 'OO (Reference)'
        ylim= 6 if structure == 'Label' else 1.7
        plot_rdf(r, gr_OO, label, legend, x_lim=r_max, y_lim=ylim, outfolder=outputFolder)
        np.savez('{}/RDF_OO.npz'.format(outputFolder), r=r, gr=gr_OO)
    
        # O-H 
        r, gr_OH = mean_rdf(samples, 'O', 'H', r_max=r_max, mic=mic, bins=bins)
        label = 'RDF_OH_{}'.format(structure)
        legend = 'OH ({})'.format(structure) if structure  !=  'P' else 'OH (Reference)'
        ylim= 60 if structure == 'Label' else 15
        plot_rdf(r, gr_OH, label, legend, x_lim=r_max, y_lim=ylim, outfolder=outputFolder)
        np.savez('{}/RDF_OH.npz'.format(outputFolder), r=r, gr=gr_OH)

        # --- ADF
        print('Calulating ADF ...')
        firstTwo = False
        onlyAngle = True
        # H-O-H: Water angle
        y_lim = 0.3 if firstTwo else 0.02
        label = "HOH_dist_{}".format(structure)
        legend = 'HOH ({})'.format(structure) if structure  !=  'P' else 'HOH (Reference)'
        angles = mean_adf(samples, 'H', 'O', 'H', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
        np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)
        plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder)
     
        # O-H-O: H-bond
        y_lim = 0.3 if firstTwo else 0.02
        label = "OHO_dist_{}".format(structure)
        legend = 'OHO ({})'.format(structure) if structure  !=  'P' else 'OHO (Reference)'
        angles = mean_adf(samples, 'O', 'H', 'O', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
        np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)
        plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder)

        # Theta of OH and z-axis 
        y_lim = 0.04
        bins = 30 
        label = "Theta_OH_dist_{}".format(structure)
        legend = "$\\theta_\\text{OH}$" + " ({})".format(structure) if structure  !=  'P' else "$\\theta_\\text{OH}$" + " (Reference)"
        angles = mean_adf_OH(samples, r_max = 1, firstTwo=False, mic=False, onlyAngle=True)
        np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)
        plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder)
