import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

showFig = True

# Function to calculate the filled area of a radar plot
def calculate_filled_area(data, angles):
    # Convert polar coordinates to Cartesian coordinates
    x = [r * np.cos(theta) for r, theta in zip(data, angles)]
    y = [r * np.sin(theta) for r, theta in zip(data, angles)]
    # Apply the Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

# Load JSON file
with open('images/similarities_testStability_Label.json', 'r') as file:
    data = json.load(file)

# Process the data
records = []
for model, metrics in data.items():
    print('Debug: model', model)
    print('Debug: metrics', metrics)
    if 'L10_L10' in model: # This is for test the stability of the model
        print('Debug: Selected model:', model)
        # Simplify model name (keeping parts containing 'C')
        #simplified_model = ''.join(part for part in model.split('_') if 'C' in part)
        #print('Debug:', simplified_model)
        
        # Extract metric values
        for metric_name, values in metrics.items():
            records.append({
                #'Model': simplified_model,
                'Model': model,
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
    ax.set_ylabel('{}'.format(metric))
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
plt.savefig('images/similarities_testStability_Label.png', dpi=300)
plt.show() if showFig else None
plt.close()

# Create a figure with 2x3 subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
# Set the title for the figure
fig.suptitle('Performace evaluation of V1 (L10L10)')
# In each subplot plot the mean WdistanceC for each metric
for i, metric in enumerate(min_max.keys()):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    metric_data = df[df['Metric'] == metric]
    sns.barplot(
        data=metric_data,
        x='Metric',
        y='WdistanceC',
        ax=ax, 
        errorbar=("pi", 50), capsize=.4,
        edgecolor="black",  # Outline color
        facecolor="none",  # Make bars hollow
        hatch="/",
    )
    # If there's '_dist' in the metric name, remove it
    xlabel = metric.replace('_dist', '')
    ax.set_xlabel(xlabel)
    #ax.set_ylabel('Mean WdistanceC')
    ax.set_ylabel('Wdistance')
    ax.set_xticklabels([])  # Remove x-tick labels
    # Add Wdistance3 mean line
    ax.axhline(y=metric_data['Wdistance3'].mean(), color='red', linestyle='-', label='V0')
    #ax.axhline(y=metric_data['WdistanceC'].mean(), color='green', linestyle='--', label='V1 mean')
    if i == 0:
        #ax.legend(loc='upper right')
        ax.legend(loc='upper right', ncol=2)    

# Adjust layout to avoid overlap
#plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('images/similarities_testStability_L10L10.png', dpi=600)
plt.show() if showFig else None

input('Press Enter to continue...')

print('The min and max values for WdistanceC are:')

final_scores = {}
# Data
categories = [r'OO', r'OH', r'$\angle$HOH', r'$\angle$ZOH', r'H-bond', r'OrderP']
for c in range(10):
    model = 'PPAFM2Exp_CoAll_L10_L10_Elatest_C{}'.format(c) 
    #print(data[model])
    #data_3 = [data[model][property]['wdistance3'] for property in ['OO_dist', 'OH_dist', 'HOH_dist', 'ThetaOH_dist', 'Hbonds']]
    data_3 = [1 - (data[model][property]['wdistance3'] - min_max[property]['min'])/(min_max[property]['max'] - min_max[property]['min']) for property in ['OO_dist', 'OH_dist', 'HOH_dist', 'ThetaOH_dist', 'Hbonds', 'OrderP']]
    data_c = [1 - (data[model][property]['wdistancec'] - min_max[property]['min'])/(min_max[property]['max'] - min_max[property]['min']) for property in ['OO_dist', 'OH_dist', 'HOH_dist', 'ThetaOH_dist', 'Hbonds',  'OrderP']]
    print(model)
    #print(data_3)
    print(data_c)

    num_vars = len(categories)

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
    ax.set_title(r'$(\lambda_1, \lambda_2) = $' + '({}, {}) C{}'.format(10, 10, c))

    # Add legend
    ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    #plt.savefig('images/{}/radar.png'.format(model), dpi=300)
    plt.savefig('images/{}_radar_C{}.png'.format(model, c), dpi=300)
    plt.show() if showFig else None
    plt.close()

final_scores['Ref'] = {'Overall': score_3, 'OO': data_3[0], 'OH': data_3[1], 'HOH': data_3[2], 'ThetaOH': data_3[3], 'Hbonds': data_3[4], 'sg': data_3[5], 'sk': data_3[6]} 
print(final_scores)
# Save final scores to a file
with open('images/final_scores_testStability.json', 'w') as file:
    json.dump(final_scores, file)
