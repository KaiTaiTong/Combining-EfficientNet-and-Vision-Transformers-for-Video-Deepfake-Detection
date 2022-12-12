#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=def-panos
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00

module load StdEnv/2020
module load gentoo/2020
module load python/3.7.7
module load intel/2020.1.217  cuda/11.4

source ~/projects/def-panos/alantkt/envs/local-editing-dataset/bin/activate
cd ~/projects/def-panos/alantkt/codes/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/cross-efficient-vit

python main.py --filter_type "unfiltered" --mixing_type "lip"
echo "End"