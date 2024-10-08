#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution


if __name__ == '__main__':
    baseOut = 'output'
    #os.makedirs(outputFolder, exist_ok=True)
    structures = ['Label', 'P', 'Prediction_1', 'Prediction_2', 'Prediction_3', \
                  'Prediction_a', 'Prediction_b', 'Prediction_c']
    #structures = ['Label']

     
    for structure in structures:
        # Create the output folders
        os.makedirs(os.path.join(baseOut, structure), exist_ok=True)

        # Read the samples
        # structure = structures[0]
    
        samples = read_samples_from_folder(os.path.join('Structures', structure))
        print('Calculating for structure: {}'.format(structure))
    
        # Common parameters for the plots, and output folder
        r_max = 4.0
        #r_max = 10.0
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
        legend = 'O-O ({})'.format(structure)
        ylim= 6 if structure == 'Label' else 1.7
        plot_rdf(r, gr_OO, label, legend, x_lim=r_max, y_lim=ylim, outfolder=outputFolder)
        np.savez('{}/RDF_OO.npz'.format(outputFolder), r=r, gr=gr_OO)
    
        # O-H 
        r, gr_OH = mean_rdf(samples, 'O', 'H', r_max=r_max, mic=mic, bins=bins)
        label = 'RDF_OH_{}'.format(structure)
        legend = 'O-H ({})'.format(structure)
        ylim= 60 if structure == 'Label' else 15
        plot_rdf(r, gr_OH, label, legend, x_lim=r_max, y_lim=ylim, outfolder=outputFolder)
        np.savez('{}/RDF_OH.npz'.format(outputFolder), r=r, gr=gr_OH)

    
        # --- ADF
        print('Calulating ADF ...')
        firstTwo = False
        onlyAngle = True
        
        # H-O-H 
        y_lim = 0.3 if firstTwo else 0.02
        label = "HOH_dist_{}".format(structure)
        legend = 'H-O-H ({})'.format(structure)
        angles = mean_adf(samples, 'H', 'O', 'H', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
        np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)
        plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder)
    
        # O-O-O 
        y_lim = 0.3 if firstTwo else 0.02
        label = "OOO_dist_{}".format(structure)
        legend = 'O-O-O ({})'.format(structure)
        angles = mean_adf(samples, 'O', 'O', 'O', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
        np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)
        plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder)
    

