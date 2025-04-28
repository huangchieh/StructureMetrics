# %%
import matplotlib.pyplot as plt
import numpy as np
from water import read_xyz_with_atomic_numbers
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle
import os

# %%
models = ['Ref', 'PPAFM2Exp_CoAll_L10_L10_Elatest', 'PPAFM2Exp_CoAll_L10_L0.1_Elatest', 'PPAFM2Exp_CoAll_L20_L1_Elatest']
angles = [0, 90, 180, 270]
samples = ['Ying_Jiang_1', 'Ying_Jiang_2_1', 'Ying_Jiang_2_2', 'Ying_Jiang_3', 'Ying_Jiang_5', 'Ying_Jiang_6', 'Ying_Jiang_4']
indexes = [[0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 6]] 

# %%
expImage = '../data/expPNG'
predictions = '../data/structures/predictions'
output = '../results/predictions'
if not os.path.exists(output):
    os.makedirs(output)

# %%
for angle in angles:
    # Look different rotations individually
    numRows = len(samples)
    numCols = len(models) + 2 # Add two for the input images
    fig, axs = plt.subplots(numRows, numCols, figsize=(numCols*1.5, numRows*1.3))
    for i, sample in enumerate(samples):
        # Load the input image: close and far 
        closeImg = '{}/{}_{}.png'.format(expImage, sample, indexes[i][0])
        farImg = '{}/{}_{}.png'.format(expImage, sample, indexes[i][1])
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
            structure = '{}/{}/Prediction_c/{}_d{}_mol.xyz'.format(predictions, model, sample, angle)
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

    #plt.tight_layout()
    plt.savefig('{}/predictions_{}.png'.format(output, angle), dpi=600)
    plt.savefig('{}/predictions_{}.pdf'.format(output, angle))
    plt.savefig('{}/predictions_{}.svg'.format(output, angle))
    plt.show()

# %%
