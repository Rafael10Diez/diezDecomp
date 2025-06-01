#!/bin/bash -l
#SBATCH -J test
#SBATCH -o %x-j%j.out
#SBATCH  --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --account=project_[please_write_project_number]
#SBATCH --partition=standard-g
#SBATCH --time=0-00:10:00
#SBATCH --exclusive

echo "Starting job $SLURM_JOB_ID at `date`"
echo "USER: ${USER}"

# ------------------------------- working directory -------------------------------
# please adjust working directory
root_dir=/users/$USER/diezDecomp-main/examples

# ------------------------------ transpose ------------------------------

cd $root_dir/halo/example_1
for my_fname in $root_dir/halo/example_1/trials/* ; do
  nproc=$(head -1  $my_fname | grep -o "[0-9]")
  echo "Running on CPU: ${my_fname}"
  srun  -n $nproc ./example_1_halo_cpu $(basename $my_fname)
  echo "Running on GPU: ${my_fname}"
  srun  -n $nproc ./example_1_halo_gpu $(basename $my_fname)
done

cd $root_dir/transpose/example_1
nproc=8
echo "Running on GPU: ${trial_subfolder}"
srun  -n $nproc ./example_1_transp_xy_8_gpu
echo "Running on CPU: ${trial_subfolder}"
srun  -n $nproc ./example_1_transp_xy_8_cpu


cd $root_dir/transpose/example_2
for my_fname in $root_dir/transpose/example_2/trials/* ; do
  nproc=$(head -1  $my_fname | grep -o "[0-9]")
  echo "Running on CPU: ${my_fname}"
  srun  -n $nproc ./example_2_transp_generalized_cpu $(basename $my_fname)
  echo "Running on GPU: ${my_fname}"
  srun  -n $nproc ./example_2_transp_generalized_gpu $(basename $my_fname)
done

cd $root_dir/transpose/example_3
for my_fname in $root_dir/transpose/example_3/trials/* ; do
  nproc=$(head -1  $my_fname | grep -o "[0-9]")
  echo "Running on CPU: ${my_fname}"
  srun  -n $nproc ./example_3_transp_generalized_cpu $(basename $my_fname)
  echo "Running on GPU: ${my_fname}"
  srun  -n $nproc ./example_3_transp_generalized_gpu $(basename $my_fname)
done
