#!/bin/bash                                          
#SBATCH --gres=gpu:1                                 
#SBATCH --constraint="32gb|40gb|48gb"
#SBATCH -c 8                                  
#SBATCH --mem=32G                                    
#SBATCH -t 08:00:00                                 
#SBATCH --output slurm/%j.out
#SBATCH --error slurm/%j.err
#SBATCH --mail-user=vincentmillions@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


#module load python/3.8
# conda activate cphoto

python $1 
