#!/bin/bash --login
#$ -cwd
#$ -o logs
#$ -j y
#$ -V
#$ -t 1-500

iteration=$SGE_TASK_ID 

module load apps/anaconda3/5.2.0/bin
source activate /mnt/jw01-aruk-home01/projects/precision_medicine_ml/tensorflow2.2_gpu

python ../run_bootstrap_evaluation.py ${das_type} ${treatment} ${iteration} ${outcome:-"class_bin"} ${type:-"log"}