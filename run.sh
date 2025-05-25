#! /bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=airawatp   # GPU partition
#SBATCH --gres=gpu:A100-SXM4:1 # Request 1 A100 GPU
#SBATCH --time=07:00:00        # Job runs for 10 minutes
#SBATCH --error=outputs/job.%J.err     # Error log (J = job ID)
#SBATCH --output=outputs/job.%J.out    # Output log

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes." 
echo "Running $SLURM_NTASKS tasks."
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_for_skt_better.py