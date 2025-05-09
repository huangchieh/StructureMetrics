from ase.io import write, read
from ase.optimize import BFGS
from gpaw import GPAW, PW

# Load geometry
atoms = read("tetrahedral_water.xyz")

# Add vacuum and turn off periodic boundary conditions
atoms.center(vacuum=5.0)
atoms.set_pbc(False)

# Set GPAW calculator
atoms.calc = GPAW(mode=PW(350), xc='PBE', txt='tetrahedral_water.log', convergence={'energy': 1e-5, 'density': 1e-4})

# Relaxation
opt = BFGS(atoms)
opt.run(fmax=0.05)

# Save result
write("tetrahedral_water_relaxed_dft.xyz", atoms)
