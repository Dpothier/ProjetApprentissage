#!/bin/bash
#PBS -N TCN
#PBS -A bue-543-aa
#PBS -l walltime=4:00:00
#PBS -l nodes=1:gpus=2
cd "${PBS_O_WORKDIR}"

module load apps/python/3.5.0

source deepenv/bin/activate

python evaluate_results_main.py gpu
python experiment_causal_soft.py gpu