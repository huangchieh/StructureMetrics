#!/usr/bin/env python
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle
from water import read_xyz_with_atomic_numbers
from water import calculate_lattice_vectors
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio.v3 as iio  

from utils import get_scan_window_from_xyz, plot_image_stack
# Input structure for visualization
demoIndex = 1
demoStructure = f'../data/structures/simulations/Label/{demoIndex}.xyz'
refZ0 = 14.0 # Needs to be confirmed latter
dz = 0.4 # Units: Å

# Output folder for the output structures
figure_folder = '../results/stacks'
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

# Parameters
sw=((0, 0, 0), (31.875, 31.875, 2.4))
ss = (25.6, 25.6, 2.4)
sw_x, sw_y, sw_z = sw[1][0], sw[1][0], sw[1][2] 
show=False
showScanRegion = False
showImageRegion = False
showLattice = False
showIndicator = False

simcolor = '#ed9d2c'
expcolor = '#de461c'
bg07color = '#479FB1'
bv17color = '#6E7CBC'

plt.rcParams['font.size']=14
#plt.rcParams['font.family']='Arial'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = True # Render text with LaTeX


xyz_min, xyz_max = get_scan_window_from_xyz(demoStructure)
xyz_center = xyz_min + (xyz_max - xyz_min)/2

####################################################################
# 1. Plot the atoms of demonstration configuration in the xy plane
####################################################################
atoms = read_xyz_with_atomic_numbers(demoStructure)

substrate = atoms[atoms.numbers == 79] # Au substrate
subPositions = substrate.get_positions()
lattice_vectors = calculate_lattice_vectors(substrate)
atoms.set_cell(lattice_vectors)
atoms.set_pbc([True, True, False])
print('lattice vector:', lattice_vectors)

repNum = (3, 3, 1)
supercell = atoms.repeat(repNum)
#view(supercell)
fig = plt.figure(figsize=(2, 2))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect('equal')
ax1.tick_params(axis='both', direction='in', labelright=False)

# Move the center
xyz_center = xyz_center + lattice_vectors[0] + lattice_vectors[1] #+ (repNum[2]-1)*lattice_vectors[2]
xy_center = xyz_center[:2]

supercellList = sorted(supercell, key=lambda atom: atom.position[2])
h_flag, o_flag, au_flag = 'H', 'O', 'Au'
for atom in supercellList:
    color = jmol_colors[atom.number]
    radius = radii[atom.number]
    if atom.number == 1:
        circle = Circle((atom.x, atom.y), radius, facecolor=color,
                        edgecolor='k', linewidth=0.5)
    elif atom.number == 8:
        circle = Circle((atom.x, atom.y), radius, facecolor=color, edgecolor='k', linewidth=0.5)
    else:
        circle = Circle((atom.x, atom.y), radius, facecolor=color, alpha=0.5)
    ax1.add_patch(circle)
xy_origin = subPositions[:, :2].min(axis=0)

ax1.set_yticks([])
ax1.set_xticks([])

offset = -3.
ax1.set_xlim([xy_center[0]-sw_x/2-offset, xy_center[0]+sw_x/2+offset])
ax1.set_ylim([xy_center[1]-sw_y/2-offset, xy_center[1]+sw_y/2+offset])

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
if show: plt.show() 

fig.savefig("{}/atoms.png".format(figure_folder), dpi=600) 
fig.savefig("{}/atoms.pdf".format(figure_folder))  
fig.savefig("{}/atoms.svg".format(figure_folder))
plt.close(fig)

########################################################
# 2. Plot simulation and exp. image stacks
########################################################
# A simulation stack
simulationInput = '../data/overview'
imagePaths = ['{}/PPAFM/{:.2f}.png'.format(simulationInput, 12+i * 0.1 * dz) for i in reversed(range(4))]
images = [np.rot90(iio.imread(imagePath).astype(np.float32), k=3) for imagePath in imagePaths]
plot_image_stack(images, figure_folder, filename_prefix="simulation_stack", show=show)

# A experimental stack
expInput = '../data/expPNG'
step = 2
imagePaths = ['{}/Ying_Jiang_2_2_{}.png'.format(expInput, i*step) for i in reversed(range(4))]
images = [np.rot90(iio.imread(imagePath).astype(np.float32), k=3) for imagePath in imagePaths]
plot_image_stack(images, figure_folder, filename_prefix="exp_stack", show=show)