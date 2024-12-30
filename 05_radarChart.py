import numpy as np
import matplotlib.pyplot as plt

# Data
categories = [r'$g_\text{OO}$', r'$g_\text{OH}$', r'$\angle$HOH', r'$\angle$ZOH', r'H-bond', r'Others']
data = {
    'Set 1': [90, 96, 94, 80, 86, 82],
    'Set 2': [85, 90, 92, 75, 80, 78],
    'Set 3': [88, 85, 95, 82, 83, 84],
}

# Number of variables
num_vars = len(categories)

# Compute angles for the categories
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the circle

# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Loop through the datasets and plot each
for label, values in data.items():
    values += values[:1]  # Close the circle
    ax.fill(angles, values, alpha=0.4, label=label)
    ax.plot(angles, values, linewidth=2)

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)

# Remove radial ticks
ax.set_yticks([])

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.show()
