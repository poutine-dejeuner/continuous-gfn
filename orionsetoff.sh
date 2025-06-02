#!/bin/bash
#SBATCH --array=1-50
#SBATCH --gres=gpu:40gb
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 09:00:00                                 
#SBATCH --output slurm/%j.out
#SBATCH --error slurm/%j.err
#SBATCH --mail-user=vincentmillions@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --comment="continuous gfn"


#module load python/3.8
# conda activate cphoto
orion hunt -n gflow --exp-max-trials 50 python tuto2.py\
--lr~'loguniform(1e-2, 1.0)' \

