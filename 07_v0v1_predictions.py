"""
import matplotlib.pyplot as plt 
import numpy as np
from water import read_xyz_with_atomic_numbers

models = ['Ref', 'PPAFM2Exp_CoAll_L10_L10_Elatest']
angles = [0, 90, 180, 270]
samples = ['Chen_CO', 'Ying_Jiang_1', 'Ying_Jiang_2_1', 'Ying_Jiang_2_2', 'Ying_Jiang_3', 'Ying_Jiang_4', 'Ying_Jiang_5', 'Ying_Jiang_6', 'Ying_Jiang_7']
folder = 'BatchOutStructures'

for angle in angles:
    # Look different rotations individually
    numRows = len(samples)
    numCols = len(models)
    fig, axs = plt.subplots(numRows, numCols, figsize=(numCols*3, numRows*3))
    for i, sample in enumerate(samples):
        for j, model in enumerate(models):
            structure = '{}/{}/Prediction_c/{}_d{}_mol.xyz'.format(folder, model, sample, angle, angle)
            atoms = read_xyz_with_atomic_numbers(structure)
            x = [atom.position[0] for atom in atoms]
            y = [atom.position[1] for atom in atoms]
            z = [atom.position[2] for atom in atoms]
            axs[i, j].scatter(x, y, c=z, cmap='viridis')
            #axs[i, j].set_title('{} {}'.format(sample, model))
            axs[i, j].set_aspect('equal')
            axs[i, j].set_xlim([min(x) - 3, max(x) + 3])
            axs[i, j].set_ylim([min(y) - 3, max(y) + 3])
            #axs[i, j].set_xlabel(r'$x$ (Å)')
            #axs[i, j].set_ylabel(r'$y$ (Å)')
    plt.tight_layout()
    plt.show() 
"""

import matplotlib.pyplot as plt
import numpy as np
from water import read_xyz_with_atomic_numbers
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle

models = ['Ref', 'PPAFM2Exp_CoAll_L10_L10_Elatest', 'PPAFM2Exp_CoAll_L10_L0.1_Elatest', 'PPAFM2Exp_CoAll_L20_L1_Elatest', 'PPAFM2Exp_CoAll_L30_L0.1_Elatest', 'PPAFM2Exp_CoAll_L40_L0.1_Elatest']
angles = [0, 90, 180, 270]
samples = ['Ying_Jiang_7', 'Chen_CO', 'Ying_Jiang_1', 'Ying_Jiang_2_1', 'Ying_Jiang_2_2', 'Ying_Jiang_3', 'Ying_Jiang_5', 'Ying_Jiang_6', 'Ying_Jiang_4']
indexes = [[0, 8], [15, 20], [0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 6]] 
folder = 'BatchOutStructures'

for angle in angles:
    # Look different rotations individually
    numRows = len(samples)
    numCols = len(models) + 2 # Add two for the input images
    fig, axs = plt.subplots(numRows, numCols, figsize=(numCols*1.5, numRows*1.3))
    for i, sample in enumerate(samples):
        # Load the input image: close and far 
        closeImg = '{}/expPNG/{}_{}.png'.format(folder, sample, indexes[i][0])
        farImg = '{}/expPNG/{}_{}.png'.format(folder, sample, indexes[i][1])
        close = plt.imread(closeImg, )
        far = plt.imread(farImg)
        # Rotate the images
        close = np.rot90(close, k=(angle+90)//90)
        far = np.rot90(far, k=(angle+90)//90)
        # Show the image with corresponding rotation angle in gray scale
        axs[i, 0].imshow(close, cmap='gray')
        axs[i, 1].imshow(far, cmap='gray')
        # Show no axis
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        for j, model in enumerate(models):
            structure = '{}/{}/Prediction_c/{}_d{}_mol.xyz'.format(folder, model, sample, angle)
            atoms = read_xyz_with_atomic_numbers(structure)


            axs[i, j+2].set_aspect('equal')
            axs[i, j+2].tick_params(axis='both', direction='in', labelright=False)

            # Sort atoms by z-position to draw farther atoms first
            atoms = sorted(atoms, key=lambda atom: atom.position[2])

            for atom in atoms:
                color = jmol_colors[atom.number]
                radius = radii[atom.number]
                circle = Circle((atom.position[0], atom.position[1]), radius, facecolor=color,
                                edgecolor='k', linewidth=0.5)
                axs[i, j+2].add_patch(circle)

            x_positions = [atom.position[0] for atom in atoms]
            y_positions = [atom.position[1] for atom in atoms]
            if j == 0:
                xmin, xmax = min(x_positions), max(x_positions)
                ymin, ymax = min(y_positions), max(y_positions)
            offset = 1
            axs[i, j+2].set_xlim([xmin - 3*offset, xmax + 3*offset])
            axs[i, j+2].set_ylim([ymin - 2*offset, ymax + 2*offset])
            # if j == 0:
            #     axs[i, j].set_ylabel(r'$y$ (Å)')
            # if i == numRows - 1:
            #     axs[i, j].set_xlabel(r'$x$ (Å)')

            #axs[i, j].set_title('{} {}'.format(sample, model))

    plt.tight_layout()
    plt.savefig('Figures/v0v1_predictions_{}.png'.format(angle), dpi=300)
    plt.savefig('Figures/v0v1_predictions_{}.pdf'.format(angle))
    plt.show()
