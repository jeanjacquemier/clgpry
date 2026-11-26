#!/usr/bin/bash
#SBATCH --job-name=cosmosis_gpry
#SBATCH --time=24:00:00
#SBATCH --ntasks=30
#SBATCH --partition=hpc,lsst
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000      # ~80 Go au total ; ajustez selon vos besoins
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

echo "---> conda activate firecrown_clp"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate firecrown_clp

# un seul thread par processus MPI
export HDF5_DO_MPI_FILE_SYNC=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "---> Allocation : ${SLURM_NTASKS} tâches, nœuds : ${SLURM_JOB_NODELIST}"

#srun -n "${SLURM_NTASKS}" bash -c 'echo "$(hostname) : $(lscpu | grep "Model name" | cut -d: -f2 | xargs)"' | sort | uniq -c
mpirun --bind-to none -n "${SLURM_NTASKS}" \
    clgpry-run --priors-path ./data/priors_file.ini  \
   --std-path  /sps/lsst/users/jacquem/miniforge3/envs/firecrown_clp/cosmosis-standard-library \
   --likelihood-source ./data/likelihood_file.py \
   --data_sacc_path ./data/clusters_sacc_file_cov.sacc 



