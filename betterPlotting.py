#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution


if __name__ == '__main__':
    baseOut = 'output'
    structures = ['Prediction_1', 'Prediction_2', 'Prediction_3', \
                  'Prediction_a', 'Prediction_b', 'Prediction_c']

    for structure in structures:
        # Create the output folders
        os.makedirs(os.path.join(baseOut, structure), exist_ok=True)
        samples = read_samples_from_folder(os.path.join('Structures', structure))
        print('Calculating for structure: {}'.format(structure))
    
        # Common parameters for the plots, and output folder
        r_max = 4.0
        mic = False
        bins = 120
        color = '#299035'
        outputFolder = os.path.join(baseOut, structure)
    
        # --- RDF 
        print('Calulating RDF ...')

        # Load Reference RDF
        ref_structure = 'P'
        ref_outputFolder = os.path.join(baseOut, ref_structure)
        ref_data = np.load('{}/RDF_OO_{}.npz'.format(ref_outputFolder, ref_structure)) 
        ref_r, ref_gr_OO = ref_data['r'], ref_data['gr']

        # O-O 
        if os.path.exists('{}/RDF_OO_{}.npz'.format(outputFolder, structure)):
            print('Loading RDF_OO_{}.npz'.format(structure))
            data = np.load('{}/RDF_OO_{}.npz'.format(outputFolder, structure))
            r, gr_OO = data['r'], data['gr']
        else:
            r, gr_OO = mean_rdf(samples, 'O', 'O', r_max=r_max, mic=mic, bins=bins)
            np.savez('{}/RDF_OO_{}.npz'.format(outputFolder, structure), r=r, gr=gr_OO)
        label = 'RDF_OO_{}'.format(structure)
        legend = 'O-O ({})'.format(structure)
        ylim= 6 if structure == 'Label' else 1.7
        # Check if ref_r and r are the same
        if np.allclose(ref_r, r):
            print('Plotting RDF_OO_{}'.format(structure))
            plot_rdf(r, [ref_gr_OO, gr_OO], label,  legend=['Reference', legend], color=['#299035', '#fc0006'], x_lim=r_max, y_lim=ylim, outfolder=outputFolder, style=['bar', 'step'])
        else:
            # Raise an error
            print('Error: Reference and current RDF r values are not the same')
            exit(1)

        # O-H 
        ref_data = np.load('{}/RDF_OH.npz'.format(ref_outputFolder))
        ref_r, ref_gr_OH = ref_data['r'], ref_data['gr']
        if os.path.exists('{}/RDF_OH.npz'.format(outputFolder, structure)):
            data = np.load('{}/RDF_OH.npz'.format(outputFolder, structure))
            r, gr_OH = data['r'], data['gr']
        else:
            r, gr_OH = mean_rdf(samples, 'O', 'H', r_max=r_max, mic=mic, bins=bins)
            np.savez('{}/RDF_OH.npz'.format(outputFolder, structure), r=r, gr=gr_OH)
        label = 'RDF_OH_{}'.format(structure)
        legend = 'OH ({})'.format(structure)
        ylim= 60 if structure == 'Label' else 15
        plot_rdf(r, [ref_gr_OH, gr_OH], label, legend=['Reference', legend], color=['#299035', '#fc0006'], x_lim=r_max, y_lim=ylim, outfolder=outputFolder, style=['bar', 'step'])
        #plot_rdf(r, gr_OH, label, legend, x_lim=r_max, y_lim=ylim, outfolder=outputFolder)
    
        # --- ADF
        print('Calulating ADF ...')
        firstTwo = False
        onlyAngle = True
        
        # H-O-H 
        ref_data = np.load('{}/HOH_dist_{}.npz'.format(ref_outputFolder, ref_structure))
        ref_angles = ref_data['angles']
        y_lim = 0.3 if firstTwo else 0.02
        label = "HOH_dist_{}".format(structure)
        legend = 'HOH ({})'.format(structure)
        if os.path.exists('{}/{}.npz'.format(outputFolder, label)):
            print('Loading HOH_dist_{}.npz'.format(structure))
            data = np.load('{}/{}.npz'.format(outputFolder, label))
            angles = data['angles']
        else:
            angles = mean_adf(samples, 'H', 'O', 'H', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
            np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)

        plot_angle_distribution([ref_angles, angles], label, legend=['Reference', legend], color=['#299035', '#fc0006'], bins=bins, y_lim=y_lim, outfolder=outputFolder, style=['bar', 'step'])
        #plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder)
    
        # O-O-O 
        ref_data = np.load('{}/OOO_dist_{}.npz'.format(ref_outputFolder, ref_structure))
        ref_angles = ref_data['angles']
        y_lim = 0.3 if firstTwo else 0.02
        label = "OOO_dist_{}".format(structure)
        legend = 'OOO ({})'.format(structure)
        if os.path.exists('{}/{}.npz'.format(outputFolder, label)):
            print('Loading OOO_dist_{}.npz'.format(structure))
            data = np.load('{}/{}.npz'.format(outputFolder, label))
            angles = data['angles']
        else:
            angles = mean_adf(samples, 'O', 'O', 'O', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
            np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)
        plot_angle_distribution([ref_angles, angles], label, legend=['Reference', legend], color=['#299035', '#fc0006'], bins=bins, y_lim=y_lim, outfolder=outputFolder, style=['bar', 'step'])
        #plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder)
    
