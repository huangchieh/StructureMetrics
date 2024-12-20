#!/usr/bin/env python

# 3D render figures from viewer's perspective x, y, z
from water import read_xyz_with_atomic_numbers
from ase.io import read, write
import os

demoStructure = 'BatchOutStructures/Label/0.xyz'
atoms = read_xyz_with_atomic_numbers(demoStructure)

# Define the perspectives
perspectives = [
    ('x', '270x,270y,0z'),  # View along x-axis
    ('y', '90x,0y,180z'),  # View along y-axis
    ('z', '0x,0y,0z')   # View along z-axis
]

# Base POV-Ray settings
povray_settings = {
    'display': False,  # Display while rendering
    'pause': True,  # Pause when done rendering (only if display)
    'transparent': True,  # Transparent background
    'canvas_width': 2500,  # Reduced width of canvas in pixels
    'camera_dist': 50.,  # Distance from camera to front atom
    'image_plane': None,  # Distance from front atom to image plane
    'point_lights': [[(10, 10, 10), 'White shadowless']],  # Simplified lighting
    'area_light': [(10, 10, 50), 'White shadowless', 2, 2, 4, 4],
    'background': 'White',  # Background color
    'textures': None,  # Length of atoms list of texture names
    'celllinewidth': 0.1,  # Radius of the cylinders representing the cell
}

# Generate images from different perspectives
for name, rotation in perspectives:
    generic_projection_settings = {
        'rotation': rotation,  # Set the specific rotation
        'radii': 0.85,  # Atom radii
        'colors': None,  # List: one (r, g, b) tuple per atom
        'show_unit_cell': 2,  # Show all of the cell
    }
    
    # Write the .pov file
    pov_file = f'atoms_{name}.pov'
    png_file = f'atoms_{name}.png'
    renderer = write(pov_file, atoms, **generic_projection_settings, povray_settings=povray_settings)
    renderer.render()


# O position distribution on z axis


# RDF ADF HBonding
