#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH -J StructureMetrics
#SBATCH -o StructureMetrics.out
module load mamba 
export OMP_NUM_THREADS=1

# Run 
python loadData.py
python main.py

# Clean up
rm -rf __pycache__/
