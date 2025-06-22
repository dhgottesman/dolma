#! /bin/sh
#SBATCH --job-name=tokenize_wikipedia_vsl_no_special
#SBATCH --output=tokenize_wikipedia_vsl_no_special.out
#SBATCH --error=tokenize_wikipedia_vsl_no_special.err
#SBATCH --time=1-00:00:00
#SBATCH --gpus=0               # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1             # Only 1 task needed
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=68  # Allocate 4 CPUs per GPU task
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable

export PYTHONPATH="$PWD"
echo $PYTHONPATH

python -m dolma.cli.main --c dolma/configs/kas.yaml tokens
