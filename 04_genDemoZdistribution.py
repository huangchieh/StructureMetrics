#!/usr/bin/env python
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle
from water import read_xyz_with_atomic_numbers
from water import read_samples_from_folder
import matplotlib.pyplot as plt
import seaborn as sns
from ase.visualize import view
from ase.io import read, write
import os

# Create the folder for the output structures
figure_folder = 'Figures'
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)


# Get the z distribution for all the structures in the Label folder
samples = read_samples_from_folder('BatchOutStructures/Label')
z = []
for structure in samples:
    atoms = read_xyz_with_atomic_numbers(structure)
    z_positions_Au = [atom.position[2] for atom in atoms if atom.symbol == 'Au']
    if len(z_positions_Au) > 0:
        mean_z_Au = sum(z_positions_Au) / len(z_positions_Au)
    else:
        mean_z_Au = 0  # or handle this case appropriately
        print('No Au atoms in {}'.format(structure))
    mean_z_Au = sum(z_positions_Au) / len(z_positions_Au)
    z_positions_O = [atom.position[2] - mean_z_Au for atom in atoms if atom.symbol == 'O']
    z.extend(z_positions_O)


# Plot the atoms of demostration configuration in the xy plane
demoStructure = 'BatchOutStructures/Label/0.xyz'
atoms = read_xyz_with_atomic_numbers(demoStructure)

fig = plt.figure(figsize=(9, 9))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect('equal')
ax1.tick_params(axis='both', direction='in', labelright=False)

atoms = sorted(atoms, key=lambda atom: atom.position[2])
for atom in atoms:
    color = jmol_colors[atom.number]
    radius = radii[atom.number]
    circle = Circle((atom.x, atom.y), radius, facecolor=color,
                            edgecolor='k', linewidth=0.5)
    ax1.add_patch(circle)

x_positions = [atom.position[0] for atom in atoms]
y_positions = [atom.position[1] for atom in atoms]
xmin, xmax = min(x_positions), max(x_positions)
ymin, ymax = min(y_positions), max(y_positions)
offset = 1
ax1.set_xlim([xmin - 3*offset, xmax + 3*offset])
ax1.set_ylim([ymin - 2*offset, ymax + 2*offset])
ax1.set_xlabel(r'$x$ (Å)')
ax1.set_ylabel(r'$y$ (Å)')
fig.subplots_adjust(hspace=0, wspace=0, left=0.08, bottom=0.15, right=0.99, top=0.95)
plt.show()
fig.savefig("{}/xy_view.png".format(figure_folder), dpi=300, bbox_inches='tight')  # Set DPI to 300
fig.savefig("{}/xy_view.pdf".format(figure_folder))  # Set DPI to 300
fig.savefig("{}/xy_view.svg".format(figure_folder))
plt.close(fig)

# Plot the atoms of demostration configuration  of cross section in the yz plane
# And the distribution of z positions of O atoms
# Use the mean z position of Au atoms as the reference plane z=0
atoms = read_xyz_with_atomic_numbers(demoStructure)
z_positions_Au = [atom.position[2] for atom in atoms if atom.symbol == 'Au']
mean_z_Au = sum(z_positions_Au) / len(z_positions_Au)
for atom in atoms:
    atom.position[2] -= mean_z_Au
z_positions_O = [atom.position[2] for atom in atoms if atom.symbol == 'O']

# Rotate view
atoms.rotate(270, 'x',  rotate_cell=True)
atoms.rotate(270, 'y', rotate_cell=True)

fig = plt.figure(figsize=(9, 3))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
ymin, ymax = -2, 12 

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect('equal')

# Get the minimum and maximum x and y positions of the atoms
x_positions = [atom.position[0] for atom in atoms]
y_positions = [atom.position[1] for atom in atoms]
xmin, xmax = min(x_positions), max(x_positions)
ymin, ymax = min(y_positions), max(y_positions)

offset = 1
xmin, xmax = xmin - 3*offset, xmax + 3*offset
ymin, ymax = ymin - 2*offset, ymax + 5*offset

ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])
ax1.tick_params(axis='both', direction='in', labelright=False)
ax1.set_xlabel(r'$y$ (Å)')
ax1.set_ylabel(r'$z$ (Å)')


# Add the atoms to the plot as circles.
# Reorder the atoms based on the z position so that the atoms at the back are plotted first
atoms = sorted(atoms, key=lambda atom: atom.position[2])
for atom in atoms:
    color = jmol_colors[atom.number]
    radius = radii[atom.number]
    circle = Circle((atom.x, atom.y), radius, facecolor=color,
                            edgecolor='k', linewidth=0.5)
    ax1.add_patch(circle)

# Plot the distribution of z positions
ax2 = fig.add_subplot(gs[0, 1])
#ax2.hist(z_positions_O, orientation='horizontal', bins=30, density=True, color=jmol_colors[8], alpha=1)
sns.kdeplot(y=z, fill=False, bw_adjust=1, ax=ax2, color=jmol_colors[8], linestyle='dotted')
sns.kdeplot(y=z_positions_O, fill=False, bw_adjust=1, ax=ax2, color=jmol_colors[8])
ax2.axhline(0, color=jmol_colors[79], linestyle='-') # Au surface
ax2.axhline(5.89, color='k', linestyle='--', lw=0.5) # 
ax2.axhline(4.85, color='k', linestyle='--', lw=0.5) # 
ax2.axhline(3.37, color='k', linestyle='--', lw=0.5) # 
ax2.set_xlabel('')
# Hide the y-axis labels on the second plot
ax2.tick_params(axis='y', labelleft=True)
ax2.set_xlabel(r'Density $\rho(z)$')
ax2.set_xlim([0, 1.2])
ax2.set_ylim([ymin, ymax])
ax2.tick_params(axis='both', direction='in', labelleft=False)
fig.subplots_adjust(hspace=0, wspace=0, left=0.05, bottom=0.15, right=0.99, top=0.95)
plt.show()
fig.savefig("{}/z_distribution.png".format(figure_folder), dpi=300, bbox_inches='tight')  # Set DPI to 300
fig.savefig("{}/z_distribution.pdf".format(figure_folder))  # Set DPI to 300
fig.savefig("{}/z_distribution.svg".format(figure_folder))
plt.close(fig)

