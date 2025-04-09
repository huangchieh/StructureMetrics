#!/usr/bin/env python
import os, json
import numpy as np
from water import sinkhorn_2d_distance
from scipy.stats import wasserstein_distance

def write_similarity_to_file(file_path, similarities):
    with open(file_path, 'w') as f:
        json.dump(similarities, f, indent=4)

if __name__ == '__main__':
    inputFolder = '../processed_data/structure_properties'
    outputFolder = '../processed_data/distribution_distances'
    os.makedirs(outputFolder, exist_ok=True)
    # Use Label or P as the target answers 
    for ground_truth in ['Label', 'P']:
        results_file = os.path.join(outputFolder, 'similarities_{}.json'.format(ground_truth))
        structures = [f for f in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, f))]
    
        all_similarities = {}
        for structure in structures:
            print('Calculating for structure: {}'.format(structure))
            similarities = {}
    
            # O-O
            data = np.load('{}/{}/OO_distances.npz'.format(inputFolder, ground_truth))
            distances = data['distances']
            datac = np.load('{}/{}/OO_distances.npz'.format(inputFolder, structure))
            distancesc = datac['distances']
            wdistancec = wasserstein_distance(distances, distancesc)
            similarities['OO_dist'] = {'wdistancec': wdistancec}
    
            # O-H
            data = np.load('{}/{}/OH_distances.npz'.format(inputFolder, ground_truth))
            distances = data['distances']
            datac = np.load('{}/{}/OH_distances.npz'.format(inputFolder, structure))
            distancesc = datac['distances']
            wdistancec = wasserstein_distance(distances, distancesc)
            similarities['OH_dist'] = {'wdistancec': wdistancec}
    
            # H-O-H 
            data = np.load('{}/{}/HOH_dist_{}.npz'.format(inputFolder, ground_truth, ground_truth))
            angles = data['angles']
            datac = np.load('{}/{}/HOH_dist_{}.npz'.format(inputFolder, structure, structure))
            anglesc = datac['angles'] 
            wdistancec = wasserstein_distance(angles, anglesc)
            similarities['HOH_dist'] = {'wdistancec': wdistancec}
    
            # Theta O-H 
            data = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(inputFolder, ground_truth, ground_truth))
            angles = data['angles']
            datac = np.load('{}/{}/Theta_OH_dist_{}.npz'.format(inputFolder, structure, structure))
            anglesc = datac['angles']
            wdistancec = wasserstein_distance(angles, anglesc)
            similarities['ThetaOH_dist'] = {'wdistancec': wdistancec}
    
            # Hbond
            data = np.load('{}/{}/Hbonds.npz'.format(inputFolder, ground_truth))['distance_angle']
            datac = np.load('{}/{}/Hbonds.npz'.format(inputFolder, structure))['distance_angle']
            wdistancec  = sinkhorn_2d_distance(data, datac)
            similarities['Hbonds'] = {'wdistancec': wdistancec}
    
            # Order parameter 2d
            data = np.load('{}/{}/OrderP.npz'.format(inputFolder, ground_truth))['sg_sk']
            datac = np.load('{}/{}/OrderP.npz'.format(inputFolder, structure))['sg_sk']
            wdistancec  = sinkhorn_2d_distance(data, datac)
            similarities['OrderP'] = {'wdistancec': wdistancec}
    
            # Store similarities for the current structure
            all_similarities[structure] = similarities
    
    
        write_similarity_to_file(results_file, all_similarities)

