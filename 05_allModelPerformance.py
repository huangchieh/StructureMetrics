import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from the table
data = {
    'Label': ['V0', '(10, 0.1)', '(10, 1)', '(10, 10)', '(20, 0.1)', '(20, 1)', '(20, 10)', '(30, 0.1)',
              '(30, 1)', '(30, 10)', '(40, 0.1)', '(40, 1)', '(40, 10)', '(50, 0.1)', '(50, 1)', '(50, 10)',
              '(60, 0.1)', '(60, 1)', '(60, 10)', '(70, 0.1)', '(70, 1)', '(70, 10)', '(80, 0.1)', '(80, 1)',
              '(80, 10)', '(90, 0.1)', '(90, 1)', '(90, 10)', '(100, 0.1)', '(100, 1)', '(100, 10)'],
    'OO': [0.142, 0.105, 0.136, 0.082, 0.134, 0.086, 0.127, 0.099, 0.106, 0.118, 0.095, 0.148, 0.134, 0.125, 0.136, 
           0.157, 0.140, 0.148, 0.154, 0.130, 0.143, 0.143, 0.120, 0.121, 0.134, 0.111, 0.152, 0.128, 0.157, 0.117, 0.119],
    'OH': [0.054, 0.052, 0.055, 0.049, 0.055, 0.054, 0.052, 0.053, 0.054, 0.053, 0.052, 0.058, 0.058, 0.057, 0.054, 
           0.058, 0.057, 0.057, 0.062, 0.055, 0.060, 0.061, 0.052, 0.057, 0.061, 0.059, 0.057, 0.062, 0.060, 0.057, 0.056],
    'HOH': [8.329, 7.479, 9.478, 7.925, 8.884, 7.799, 8.417, 8.332, 8.657, 8.253, 7.678, 10.709, 11.553, 8.937, 9.471, 
            9.52, 8.11, 8.922, 10.368, 8.757, 10.216, 8.854, 8.261, 9.51, 9.529, 8.729, 8.138, 10.176, 9.158, 9.75, 10.089],
    'ZOH': [20.72, 20.194, 17.732, 19.618, 20.032, 15.991, 21.168, 20.791, 20.113, 20.465, 20.865, 15.472, 18.46, 19.102,
            18.89, 22.18, 18.653, 17.775, 18.885, 17.895, 18.065, 18.957, 21.705, 19.448, 20.091, 16.903, 20.676, 19.136, 
            19.779, 19.136, 16.719],
    'Hbond': [7.204, 2.897, 5.226, 1.463, 4.071, 2.513, 7.003, 4.252, 5.684, 7.491, 4.747, 6.725, 5.288, 5.655, 5.962,
              5.207, 9.374, 7.307, 9.366, 7.771, 7.149, 7.204, 4.498, 7.185, 6.323, 5.777, 6.143, 5.725, 7.638, 5.009, 5.836]
}

# Convert to DataFrame
df = pd.DataFrame(data)

metrics = ['OO', 'OH', 'HOH', 'ZOH', 'Hbond']
baseline = df.loc[0, metrics]  # V0 baseline values

# Create a plot for each metric
fig, axs = plt.subplots(5, 1, figsize=(6, 10), dpi=300, sharex=True)
fig.suptitle('Comparison of Structural Metrics for Different $(\lambda_1, \lambda_2)$ Values', fontsize=14)

for i, metric in enumerate(metrics):
    ax = axs[i]
    #ax.plot(df['Label'], df[metric], marker='o', label=metric, color='tab:blue')
    ax.plot(df['Label'], df[metric], marker='o', color='tab:blue')
    ax.axhline(y=baseline[metric], color='r', linestyle='--', label='Baseline (V0)')
    ax.fill_between(df['Label'], df[metric], baseline[metric], where=(df[metric] < baseline[metric]), 
                    facecolor='green', alpha=0.3, interpolate=True, label='Improvement' if i == 0 else "")
    ax.fill_between(df['Label'], df[metric], baseline[metric], where=(df[metric] > baseline[metric]), 
                    facecolor='red', alpha=0.3, interpolate=True, label='Deterioration' if i == 0 else "")

    ax.set_ylabel(metric, fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    if i == 0:
        ax.legend(loc='lower right', fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

axs[-1].set_xlabel('$(\lambda_1, \lambda_2)$ values', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.subplots_adjust(top=0.92)
plt.savefig('allModelPerformance.png', dpi=300, bbox_inches='tight')
plt.show()

