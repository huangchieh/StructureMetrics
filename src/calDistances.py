#!/usr/bin/env python
import os, json
import numpy as np
from water import sinkhorn_2d_distance
from scipy.stats import wasserstein_distance

def write_similarity_to_file(file_path, similarities):
    with open(file_path, 'w') as f:
        json.dump(similarities, f, indent=4)

if __name__ == '__main__':
    inputFolder = '../processed_data/structure_properties'  # Predictions
    theoryFolder = '../processed_data/theory_distributions' # Reference DFT 
    outputFolder = '../processed_data/distribution_distances'
    os.makedirs(outputFolder, exist_ok=True)
    for ground_truth in ['Label']: # or ['Label', 'P']
        for layer in ['All', 'Top', 'Bottom']:
            results_file = os.path.join(outputFolder, 'similarities_{}_{}.json'.format(ground_truth, layer))
            structures = [f for f in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, f))]
            all_similarities = {}
            for structure in structures:
                print('Calculating for structure: {}'.format(structure))
                similarities = {}
        
                # O-O
                data = np.load(f'{theoryFolder}/{ground_truth}/OO_{layer}.npz')
                distances = data['OO']
                datac = np.load('{}/{}/OO.npz'.format(inputFolder, structure))
                distancesc = datac['OO']
                wdistancec = wasserstein_distance(distances, distancesc)
                similarities['OO_dist'] = {'wdistancec': wdistancec}
        
                # O-H
                data = np.load(f'{theoryFolder}/{ground_truth}/OH_{layer}.npz')
                distances = data['OH']
                datac = np.load('{}/{}/OH.npz'.format(inputFolder, structure))
                distancesc = datac['OH']
                wdistancec = wasserstein_distance(distances, distancesc)
                similarities['OH_dist'] = {'wdistancec': wdistancec}
        
                # H-O-H 
                data = np.load(f'{theoryFolder}/{ground_truth}/HOH_{layer}.npz')
                angles = data['HOH']
                datac = np.load('{}/{}/HOH.npz'.format(inputFolder, structure))
                anglesc = datac['HOH'] 
                wdistancec = wasserstein_distance(angles, anglesc)
                similarities['HOH_dist'] = {'wdistancec': wdistancec}
        
                # Z-O-H 
                data = np.load(f'{theoryFolder}/{ground_truth}/ZOH_{layer}.npz')
                angles = data['ZOH']
                datac = np.load('{}/{}/ZOH.npz'.format(inputFolder, structure))
                anglesc = datac['ZOH']
                wdistancec = wasserstein_distance(angles, anglesc)
                similarities['ThetaOH_dist'] = {'wdistancec': wdistancec}
        
                # Hbond
                data = np.load(f'{theoryFolder}/{ground_truth}/Hbond_{layer}.npz')['OO_OHO']
                datac = np.load('{}/{}/Hbond.npz'.format(inputFolder, structure))['OO_OHO']
                wdistancec  = sinkhorn_2d_distance(data, datac)
                similarities['Hbonds'] = {'wdistancec': wdistancec}
        
                # Order parameter 2d
                data = np.load(f'{theoryFolder}/{ground_truth}/OrderP_{layer}.npz')['sg_sk']
                datac = np.load('{}/{}/OrderP.npz'.format(inputFolder, structure))['sg_sk']
                wdistancec  = sinkhorn_2d_distance(data, datac)
                similarities['OrderP'] = {'wdistancec': wdistancec}
        
                # Store similarities
                all_similarities[structure] = similarities
        
        
            write_similarity_to_file(results_file, all_similarities)

