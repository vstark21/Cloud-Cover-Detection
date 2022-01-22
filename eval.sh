#!/bin/bash
#PBS -l select=ncpus=8:mem=16gb:ngpus=1 
#PBS -q gpu                             

module load cuda/10
module load anaconda/3
source activate virenv

cd Cloud-Cover-Detection

python eval.py

conda deactivate
module unload anaconda/3
module unload cuda
