#!/usr/bin/env python

# 3D render figures from viewer's perspective x, y, z
from water import read_xyz_with_atomic_numbers
from ase.visualize import view
from ase.io import read, write
import os

demoStructure = 'BatchOutStructures/Label/0.xyz'
atoms = read_xyz_with_atomic_numbers(demoStructure)

# Visualize the atoms
view(atoms)

# Render the atoms and save the image
generic_projection_settings = {
    'rotation': '0x,0y,0z',  # text string with rotation (default='' )
    'radii': .85,  # float, or a list with one float per atom
    'colors': None,  # List: one (r, g, b) tuple per atom
    'show_unit_cell': 2,   # 0, 1, or 2 to not show, show, and show all of cell
}

povray_settings = {
    'display': False,  # Display while rendering
    'pause': True,  # Pause when done rendering (only if display)
    'transparent': True,  # Transparent background
    'canvas_width': 1000,  # Reduced width of canvas in pixels
    'canvas_height': None,  # Reduced width of canvas in pixels
    'camera_dist': 50.,  # Distance from camera to front atom
    'image_plane': None,  # Distance from front atom to image plane
    'point_lights': [],  # Simplified lighting
    'area_light': [(2., 3., 40.),  # location
                   'White',  # color
                   .7, .7, 3, 3],  # Reduced number of lamps
    'background': 'White',  # color
    'textures': None,  # Length of atoms list of texture names
    'celllinewidth': 0.1,  # Radius of the cylinders representing the cell
}

write('atoms.pov', atoms, **generic_projection_settings, povray_settings=povray_settings)

# Render the .pov file to an image
os.system('povray +Iatoms.pov +Oatoms.png +W800 +H800 +D +FN')


# O position distribution on z axis


# RDF ADF HBonding