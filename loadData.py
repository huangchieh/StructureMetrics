#!/usr/bin/env python3

import os
import webdataset as wds

# Specify the exact files you want to load
base = "/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-bilayer-FB/Water-bilayer_FB_Ref/"

# Same as Stage2
urls_test = "Water-bilayer-K-{1..10}_test_{0..3}.tar"
urls_train = "Water-bilayer-K-{1..10}_train_{0..31}.tar"
urls_val = "Water-bilayer-K-{1..5}_val_{0..3}.tar"

for urls, name in zip([urls_test, urls_train, urls_val], ['test', 'train', 'val']):
    print(f"Processing dataset: {urls}")
    dataset_path = base + urls
    dataset = wds.WebDataset(dataset_path)

    # Filter the dataset to include only samples that contain the 'xyz' key
    xyz_dataset = dataset.select(lambda sample: 'xyz' in sample)

    # Ensure the target directory exists
    output_dir = 'Structures/Label/{}/'.format(name)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the filtered dataset and process .xyz files
    for sample in xyz_dataset:
        if 'xyz' in sample:
            xyz_data = sample['xyz'].decode('utf-8')  # Decoding the text data from .xyz files
            file_name = sample['__key__'] + '.xyz'  # Constructing the file name
            file_path = os.path.join(output_dir, file_name)  # Full path for the file

            # Write the .xyz data to a file
            with open(file_path, 'w') as f:
                f.write(xyz_data)
        else:
            print("No 'xyz' key found in this sample:", sample['__key__'])
