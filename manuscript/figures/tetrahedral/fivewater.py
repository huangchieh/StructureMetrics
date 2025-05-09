from ase import Atoms
from ase.build import molecule
from ase.io import write
import numpy as np

def place_tetrahedral(center, distance):
    """
    Generate 4 positions around `center` forming a tetrahedron.
    """
    a = distance
    return np.array([
        [ a,  a,  a],
        [-a, -a,  a],
        [-a,  a, -a],
        [ a, -a, -a],
    ]) / np.sqrt(3) + center

# Place one central water molecule
waters = []
center = np.array([0, 0, 0])
waters.append(molecule("H2O"))
waters[0].translate(center)

# Place four surrounding waters at tetrahedral positions
positions = place_tetrahedral(center, distance=3.0)  # 3 Å is typical O-O distance
for pos in positions:
    h2o = molecule("H2O")
    h2o.translate(pos)
    waters.append(h2o)

# Combine all into one system
system = sum(waters[1:], waters[0])
write("tetrahedral_water.xyz", system)
