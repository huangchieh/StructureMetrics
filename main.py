#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples
from water import mean_rdf, mean_adf, mean_distance_distribution
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution

if __name__ == '__main__':
    # Create output folder
    if not os.path.exists('statistic'):
        os.makedirs('statistic')

    # Read the samples
    if len(sys.argv) != 3:
        print('Usage: python {}  <#folderToSee> <#xyzToSee>'.format(sys.argv[0]))
        sys.exit(1)
    toSee = (int(sys.argv[1]), int(sys.argv[2]))
    simulation = 'Water-bilayer'
    samples = read_samples(simulation, toSee)

    # Common parameters for plotting
    r_max = 10.0
    mic = True
    bins = 120
    color = '#299035'

    # --- RDF 
    print('Calulating RDF ...')
    # O-O 
    r, gr_OO = mean_rdf(samples, 'O', 'O',  mic=mic, bins=bins)
    label = 'RDF_OO_{}_{}_{}'.format(simulation, toSee[0], toSee[1])
    legend = 'O-O (Target)'
    plot_rdf(r, gr_OO, label, legend)
    np.savez('statistic/RDF_OO_{}_{}_{}.npz'.format(simulation, toSee[0], toSee[1]), r=r, gr=gr_OO)

    # O-H 
    r, gr_OH = mean_rdf(samples, 'O', 'H',  mic=mic, bins=bins)
    label = 'RDF_OH_{}_{}_{}'.format(simulation, toSee[0], toSee[1])
    legend = 'O-H (Target)'
    plot_rdf(r, gr_OH, label, legend)
    np.savez('statistic/RDF_OH_{}_{}_{}.npz'.format(simulation, toSee[0], toSee[1]), r=r, gr=gr_OH)

    # --- ADF
    print('Calulating ADF ...')
    firstTwo = False
    onlyAngle = True
    
    # H-O-H 
    y_lim = 0.3 if firstTwo else 0.02
    label = "HOH_dist_{}_FT{}_{}_{}".format(simulation, int(firstTwo), toSee[0], toSee[1])
    legend = 'H-O-H (Target)'
    angles = mean_adf(samples, 'H', 'O', 'H', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
    np.savez('statistic/{}.npz'.format(label), angles=angles)
    plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim)

    # O-O-O 
    y_lim = 0.3 if firstTwo else 0.02
    label = "OOO_dist_{}_FT{}_{}_{}".format(simulation, int(firstTwo), toSee[0], toSee[1])
    legend = 'O-O-O (Target)'
    angles = mean_adf(samples, 'O', 'O', 'O', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle)
    np.savez('statistic/{}.npz'.format(label), angles=angles)
    plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim)

    # --- Atoms distances distributions
    y_lim = 0.4
    # O-H 
    label = 'OH_dist_{}_{}_{}'.format(simulation, toSee[0], toSee[1])
    legend = 'O-H (Target)'
    distances = mean_distance_distribution(samples, 'O', 'H', mic=True, dr=0.1, r_max=r_max, onlyDistances=True)
    np.savez('statistic/{}.npz'.format(label), distances=distances)
    plot_distance_distribution(distances, label, legend=legend, r_max=r_max,  color=color, bins=bins, y_lim=y_lim)

    # O-O 
    label = 'OO_dist_{}_{}_{}'.format(simulation, toSee[0], toSee[1])
    legend = 'O-O (Target)'
    distances = mean_distance_distribution(samples, 'O', 'O', mic=True, dr=0.1, r_max=r_max, onlyDistances=True)
    np.savez('statistic/{}.npz'.format(label), distances=distances)
    plot_distance_distribution(distances, label, legend, r_max=r_max,  color=color, bins=bins, y_lim=y_lim)

