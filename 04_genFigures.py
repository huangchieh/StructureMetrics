# 3D render  figures from viwer's perspective x, y, z
from water import read_xyz_with_atomic_numbers
from ase.visualize import view

demoStructure = 'BatchOutStructures/Label/0.xyz'
atoms = read_xyz_with_atomic_numbers(demoStructure)

view(atoms)

# O position distribution on z axis


# RDF ADF HBonding