#!/bin/bash
##SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --account=def-eroussea#rrg-corbeilj-ac
#SBATCH --mail-user=sewagnouin-rogia.kpanou.1@ulaval.ca
#SBATCH --mail-type=ALL

SECONDS=0
python $@
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
