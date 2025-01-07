import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

showFig = False

# Function to calculate the filled area of a radar plot
def calculate_filled_area(data, angles):
    # Convert polar coordinates to Cartesian coordinates
    x = [r * np.cos(theta) for r, theta in zip(data, angles)]
    y = [r * np.sin(theta) for r, theta in zip(data, angles)]
    # Apply the Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

# Load JSON file
with open('images/similarities_Label.json', 'r') as file:
    data = json.load(file)

# Process the data
records = []
for model, metrics in data.items():
    simplified_model = ''.join(part for part in model.split('_') if 'L' in part)  # Simplify model name
    for metric_name, values in metrics.items():
        records.append({
            'Model': simplified_model,
            'Metric': metric_name,
            'Wdistance3': values['wdistance3'],
            'WdistanceC': values['wdistancec'],
            'WdistanceDecrease': values['wdistance_decrease']
        })

# Convert to DataFrame
df = pd.DataFrame(records)

# Unique metrics for plotting
metrics = df['Metric'].unique()

# Create subplots
fig, axes = plt.subplots(nrows=len(metrics), figsize=(20, 10), sharex=True)

min_max = {}
# Plot each metric in its own subplot
for i, metric in enumerate(metrics):
    metric_data = df[df['Metric'] == metric]
    ax = axes[i] if len(metrics) > 1 else axes  # Handle single subplot case
    sns.lineplot(
        data=metric_data,
        x='Model',
        y='Wdistance3',
        marker='o',
        label='V0',
        ax=ax
    )
    sns.lineplot(
        data=metric_data,
        x='Model',
        y='WdistanceC',
        marker='o',
        label='V1',
        ax=ax
    )
    # Mark min/max for WdistanceC
    min_wc = metric_data.loc[metric_data['WdistanceC'].idxmin()]
    max_wc = metric_data.loc[metric_data['WdistanceC'].idxmax()]
    ax.annotate('Min', (min_wc['Model'], min_wc['WdistanceC']), 
                textcoords="offset points", xytext=(-10, -10), ha='center', color='green')
    ax.annotate('Max', (max_wc['Model'], max_wc['WdistanceC']), 
                textcoords="offset points", xytext=(-10, 10), ha='center', color='green')
    
    min_max[metric] = {'min': min_wc['WdistanceC'], 'max': max_wc['WdistanceC']}

    #ax.set_title(f'{metric} - Wdistance3 and WdistanceC')
    ax.set_ylabel('WDistance {}'.format(metric))
    if i != len(metrics) - 1:
        ax.set_xlabel('')
        #ax.set_xticklabels([])  # Remove x-tick labels except for the last plot
    ax.legend(loc='upper right')

# Set x-axis label for the last subplot
axes[-1].set_xlabel('Model')
axes[-1].tick_params(axis='x', rotation=45)

# Adjust layout to avoid overlap
#plt.tight_layout()
print(min_max)
plt.savefig('images/similarities_Label.png', dpi=300)
plt.show() if showFig else None
plt.close()

print('The min and max values for WdistanceC are:')

final_scores = {}
# Data
categories = [r'OO', r'OH', r'$\angle$HOH', r'$\angle$ZOH', r'H-bond']
for L1 in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    for L2 in [0.1, 1, 10]:
        model = 'PPAFM2Exp_CoAll_L{}_L{}_Elatest'.format(L1, L2) 
        #print(data[model])
        #data_3 = [data[model][property]['wdistance3'] for property in ['OO_dist', 'OH_dist', 'HOH_dist', 'ThetaOH_dist', 'Hbonds']]
        data_3 = [1 - (data[model][property]['wdistance3'] - min_max[property]['min'])/(min_max[property]['max'] - min_max[property]['min']) for property in ['OO_dist', 'OH_dist', 'HOH_dist', 'ThetaOH_dist', 'Hbonds']]
        data_c = [1 - (data[model][property]['wdistancec'] - min_max[property]['min'])/(min_max[property]['max'] - min_max[property]['min']) for property in ['OO_dist', 'OH_dist', 'HOH_dist', 'ThetaOH_dist', 'Hbonds']]
        print(model)
        #print(data_3)
        print(data_c)

        num_vars = 5

        # Compute angles for the categories
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Loop through the datasets and plot each
        data_3.append(data_3[0])  # Close the circle
        data_c.append(data_c[0])  # Close the circle
        #ax.fill(angles, values, alpha=0.4, label=label)
        #ax.plot(angles, data_3, linewidth=1, label='V0')
        ax.fill(angles, data_3, alpha=0.4, label='V0', zorder=2)
        ax.fill(angles, data_c, alpha=0.4, label='V1', zorder=1)


        # Calculate areas for data_3 and data_c
        area = calculate_filled_area([1, 1, 1, 1, 1, 1], angles)
        area_3 = calculate_filled_area(data_3, angles)
        area_c = calculate_filled_area(data_c, angles)
        score_3 = (area_3 / area)
        score_c = (area_c / area)
        final_scores[model] = {'Overall': score_c, 'OO': data_c[0], 'OH': data_c[1], 'HOH': data_c[2], 'ThetaOH': data_c[3], 'Hbonds': data_c[4]} 
        print('Area for best:', area)
        print('Area for V0:', area_3)
        print('Area for V1:', area_c)
        print('Percentage of V0 relative to best:', (area_3 / area) * 100, '%')
        print('Percentage of V1 relative to best:', (area_c / area) * 100, '%')

        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)

        # Remove radial ticks
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        # set title
        ax.set_title(r'$(\lambda_1, \lambda_2) = $' + '({}, {})'.format(L1, L2))

        # Add legend
        ax.legend(loc='upper right', frameon=False)

        plt.tight_layout()
        plt.savefig('images/{}/radar.png'.format(model), dpi=300)
        plt.show() if showFig else None
        plt.close()

final_scores['Ref'] = {'Overall': score_3, 'OO': data_3[0], 'OH': data_3[1], 'HOH': data_3[2], 'ThetaOH': data_3[3], 'Hbonds': data_3[4]} 
print(final_scores)
# Save final scores to a file
with open('images/final_scores.json', 'w') as file:
    json.dump(final_scores, file)
