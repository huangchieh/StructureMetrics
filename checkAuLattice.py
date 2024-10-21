#!/usr/bin/env python

# This script calculates the lattice constant of a 2D Au lattice
# by calculating the distance between neighbouring atoms in the
# first peak of the RDF. Then, the 3D lattice constant is calculated
# by multiplying the 2D lattice constant by sqrt(2).

from water import read_samples_from_folder
from water import all_distances
import matplotlib.pyplot as plt
import numpy as np

# Load structure
structure = 'Structures/Label'
sample = read_samples_from_folder(structure)

r_max = 5
mic = False
bins = 120 

# Plot RDF and compare with experimental data
Au_Au_distances = all_distances(sample, 'Au', 'Au', r_max, bins, mic)
# Plot the distribution
Au_Au_distances = np.array(Au_Au_distances) 
plt.hist(Au_Au_distances, bins=bins, density=True, alpha=0.6, color='g')
plt.show()

# Neighbouring atoms located at the first peak of the previous histogram
neighbour_distances = Au_Au_distances[Au_Au_distances < 3]
lattice_constant_2d = np.mean(neighbour_distances)
print('2D lattice constant: {:.3f} Å'.format(lattice_constant_2d))
lattice_constant = lattice_constant_2d * np.sqrt(2)
print('3D lattice constant: {:.3f} Å ( Exp. 4.078 Å Ref.82 in https://doi.org/10.1063/1.5127099)'.format(lattice_constant))
