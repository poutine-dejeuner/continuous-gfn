#!/bin/bash
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

python $1 
