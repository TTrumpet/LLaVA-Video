#!/bin/bash
#SBATCH --account=mshah
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH -c 20
#SBATCH --mem-per-cpu=24G
#SBATCH --time=72:00:00
#SBATCH --job-name=LLaVA-Video_train
#SBATCH -e slurm_output-%j.err
#SBATCH -o slurm_output-%j.out

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo $SLURM_JOB_NODELIST

module load anaconda
source activate llava
module load llava
module load cuda/cuda-11.3
module load gcc

export PYTHONPATH="./:$PYTHONPATH"

### Running the program ###
nvidia-smi && srun bash scripts/video/train/SO400M_Qwen2.5_3.0B_ov_to_video_vidshikra.sh
