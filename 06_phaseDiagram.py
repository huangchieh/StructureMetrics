import json
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load JSON file
with open('images/final_scores.json', 'r') as file:
    scores = json.load(file)

# Define L1 and L2 values
L1_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
L2_values = [0.1, 1, 10]

# Create an empty grid to store the scores
score_grid = np.zeros((6, len(L2_values), len(L1_values)))
score_increase = np.zeros_like(score_grid)
labels = ['OO', 'OH', r'$\angle$HOH', r'$\angle$ZOH', 'H-bond', 'Overall']
# Populate the grid with scores
for j, L2 in enumerate(L2_values):
    for i, L1 in enumerate(L1_values):
        model = f'PPAFM2Exp_CoAll_L{L1}_L{L2}_Elatest'
        score_grid[0, j, i] = scores[model]['OO'] 
        score_grid[1, j, i] = scores[model]['OH'] 
        score_grid[2, j, i] = scores[model]['HOH'] 
        score_grid[3, j, i] = scores[model]['ThetaOH'] 
        score_grid[4, j, i] = scores[model]['Hbonds'] 
        score_grid[5, j, i] = scores[model]['Overall'] 

        # Calculate the increase in score compared to the reference model
        score_increase[0, j, i] = scores[model]['OO'] - scores['Ref']['OO']
        score_increase[1, j, i] = scores[model]['OH'] - scores['Ref']['OH']
        score_increase[2, j, i] = scores[model]['HOH'] - scores['Ref']['HOH']
        score_increase[3, j, i] = scores[model]['ThetaOH'] - scores['Ref']['ThetaOH']
        score_increase[4, j, i] = scores[model]['Hbonds'] - scores['Ref']['Hbonds']
        score_increase[5, j, i] = scores[model]['Overall'] - scores['Ref']['Overall']

def plot_heatmaps(score_data, title_suffix, cmap='viridis', filename=None):
    # Calculate global vmin and vmax for consistent color scale
    vmin = np.min(score_data)
    vmax = np.max(score_data)

    # Create subplots for 6 heatmaps in a grid
    fig, axes = plt.subplots(3, 2, figsize=(10, 5))  # 3 rows, 2 columns

    # Loop over each subplot
    for i, ax in enumerate(axes.flat):
        # Plot the heatmap with consistent vmin and vmax
        im = ax.imshow(score_data[i], cmap=cmap, aspect='equal', origin='lower', vmin=vmin, vmax=vmax)

        # Add labels and title
        ax.set_xticks(range(len(L1_values)))
        ax.set_xticklabels(L1_values, fontsize=8)
        ax.set_yticks(range(len(L2_values)))
        ax.set_yticklabels(L2_values, fontsize=8)
        ax.set_xlabel('$\lambda_{cycle}$', fontsize=10) if i > 3 else None
        ax.set_ylabel('$\lambda_{identity}$', fontsize=10) if i % 2 == 0 else None

        # Create a colorbar with matching height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r'{} score {}'.format(labels[i], title_suffix), fontsize=10)

    # Adjust layout to prevent overlap
    fig.subplots_adjust(hspace=0.1, wspace=0.3, left=0.1, bottom=0.1, right=0.90, top=0.95)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()

# Plot heatmaps for score and score_increase
plot_heatmaps(score_grid, '', filename='images/phase_diagram.png')
plot_heatmaps(score_increase, '$\Delta$', cmap='BrBG', filename='images/phase_diagram_increase.png')