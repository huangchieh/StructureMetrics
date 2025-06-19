import matplotlib.pyplot as plt
from mlspm.graph import MoleculeGraph
from mlspm.utils import read_xyzs
from ppafm.ocl.AFMulator import AFMulator
import numpy as np
from water import read_xyz_with_atomic_numbers
from ase.data.colors import jmol_colors
from ase.data import covalent_radii as radii
from matplotlib.patches import Circle
import os

def get_sim(mol, exp_data, params):
    amp = 2.0
    nx, ny, nz = exp_data['data'].shape 
    dist = params['dist'] #

    xyzs = mol.array(xyz=True)
    Zs = mol.array(element=True).astype(np.int32)[:, 0]
    qs = np.zeros(len(xyzs))

    df_steps = round(amp / 0.1)
    scan_dim = (nx, ny, df_steps + nz - 1,)
    zmin = xyzs[:, 2].max() + dist
    zmax = zmin + (amp - 0.1) + (nz - 1) * 0.1
    scan_window = ((0.01, 0.01, zmin), (exp_data['lengthX'], exp_data['lengthY'], zmax))

    afmulator = AFMulator(
        pixPerAngstrome=10,
        scan_dim=scan_dim,
        scan_window=scan_window,
        iZPP=8,
        df_steps=df_steps
    )

    X = afmulator(xyzs, Zs, qs)

    return X

params = [
        {'pred_dir': 'predictions_augmentation'  , 'exp_name': 'Ying_Jiang_1'  , 'label': 'C', 'dist': 5.1, 'offset': ( 0.0, -1.0)},
        {'pred_dir': 'predictions_augmentation'  , 'exp_name': 'Ying_Jiang_2_1', 'label': 'D', 'dist': 5.0, 'offset': ( 1.0,  0.5)},
        {'pred_dir': 'predictions_augmentation'  , 'exp_name': 'Ying_Jiang_2_2', 'label': 'E', 'dist': 4.9, 'offset': ( 0.0,  0.0)},
        {'pred_dir': 'predictions_augmentation'  , 'exp_name': 'Ying_Jiang_3'  , 'label': 'F', 'dist': 4.8, 'offset': ( 0.0, -2.0)},
        {'pred_dir': 'predictions_augmentation'  , 'exp_name': 'Ying_Jiang_5'  , 'label': 'G', 'dist': 5.0, 'offset': ( 2.0,  0.0)},
        {'pred_dir': 'predictions_augmentation'  , 'exp_name': 'Ying_Jiang_6'  , 'label': 'H', 'dist': 4.8, 'offset': ( 1.5,  2.0)},
        ]

output = '../results/ppafm4predictions'
os.makedirs(output, exist_ok=True)
samples = ['Ying_Jiang_1', 'Ying_Jiang_2_1', 'Ying_Jiang_2_2', 'Ying_Jiang_3', 'Ying_Jiang_5', 'Ying_Jiang_6'] 
indexes = [[0, 8], [0, 8], [0, 8], [0, 8], [0, 8], [0, 8]] 
models = ['Ref_best', 'PPAFM2Exp_CoAll_L20_L1_Elatest_C1', 'PPAFM2Exp_CoAll_L10_L10_Elatest_C6', 'PPAFM2Exp_CoAll_L50_L1_Elatest']
expImage = '../data/expPNG'
model_path = '../data/structures/Predictions'
numRows = len(samples)
numCols = 5

pred_bonds = None # We don't need bonds for PPAFM simulation 
classes = [[1], [8]]
angles = [0, 90, 180, 270]
angle = angles[0]
model =  models[1]
for angle in angles:
    for model in models:
        fig, axs = plt.subplots(numRows, numCols, figsize=(numCols*2, numRows*2 + 2.5))
        print(f"Visualising predictions for model {model} at angle {angle} degrees")
        for i, sample in enumerate(samples):
            # Load the input image: close and far 
            closeImg = '{}/{}_{}.png'.format(expImage, sample, indexes[i][0])
            farImg = '{}/{}_{}.png'.format(expImage, sample, indexes[i][1])
            close = plt.imread(closeImg)
            far = plt.imread(farImg)
            # Rotate the images
            close = np.rot90(close, k=(angle+90)//90)
            far = np.rot90(far, k=(angle+90)//90)
            # Show the image with corresponding rotation 
            axs[i, 0].imshow(close, cmap='inferno')
            axs[i, 1].imshow(far, cmap='inferno')
            # Show no axis
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')

            # Load the prediction 
            xyz_file = f"{model_path}/{model}/Prediction_c/{sample}_d{angle}_mol.xyz"
            atoms = read_xyz_with_atomic_numbers(xyz_file)
            axs[i, 2].set_aspect('equal')
            axs[i, 2].tick_params(axis='both', direction='in', labelright=False)
            axs[i, 2].set_xticks([])
            axs[i, 2].set_yticks([])
            #axs[i, 2].axis('off')

            atoms = sorted(atoms, key=lambda atom: atom.position[2])

            z_values = np.array([atom.position[2] for atom in atoms])
            z_min, z_max = np.min(z_values), np.max(z_values)
            z_range = z_max - z_min if z_max > z_min else 1.0  # avoid divide-by-zero


            for atom in atoms:
                color = jmol_colors[atom.number]
                radius = radii[atom.number]*1.3
                x, y, z = atom.position[0], atom.position[1], atom.position[2]
                # Scale size by z depth (closer atoms are larger)
                scale = 0.5 + 0.5 * (z - z_min) / z_range  # scale between 0.5x and 1.0x
                scaled_radius = radius * scale
                circle = Circle((x, y), scaled_radius, facecolor=color, edgecolor='k', linewidth=0.5)
                axs[i, 2].add_patch(circle)

            x_positions = [atom.position[0] for atom in atoms]
            y_positions = [atom.position[1] for atom in atoms]

            xmin, xmax = min(x_positions), max(x_positions)
            ymin, ymax = min(y_positions), max(y_positions)
            offset = 1

            # Calculate the center and the maximum span to ensure square axes
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            span = max(xmax - xmin, ymax - ymin) / 2 + 6 * offset  # add padding

            axs[i, 2].set_xlim([x_center - span, x_center + span])
            axs[i, 2].set_ylim([y_center - span, y_center + span])
            # Draw 1 nm scale bar at top left for the first model
            # 1 nm in Angstroms (10 Å)
            scale_length = 10  # 1 nm = 10 Å
            # Place bar 5% from left and 10% from top
            x0 = x_center - span + 0.05 * (2 * span)
            y0 = y_center + span - 0.98 * (2 * span)
            x1 = x0 + scale_length
            y1 = y0
            axs[i, 2].plot([x0, x1], [y0, y1], color='k')
            axs[i, 2].text((x0 + x1) / 2, y0 + 0.03 * (2 * span), '1 nm', color='k',
                        ha='center', va='bottom', fontsize=10)
            # Latter 
            pred_xyz = read_xyzs([xyz_file])[0]
            mol = MoleculeGraph(pred_xyz, pred_bonds, classes=classes)
            exp_data = np.load(f"../data/expNpz/{sample}.npz") # To load height data
            X = get_sim(mol, exp_data, params[i])
            close_pred = X[:, :, 0].T
            far_pred = X[:, :, -1].T
            axs[i, 3].imshow(close_pred, origin='lower', cmap='inferno')
            axs[i, 4].imshow(far_pred, origin='lower', cmap='inferno')
            axs[i, 3].axis('off')
            axs[i, 4].axis('off')
        plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.025, right=0.975)
        plt.savefig(f"{output}/ppafm4predictions_{model}_{angle}.svg")
        plt.show()
        plt.close(fig)