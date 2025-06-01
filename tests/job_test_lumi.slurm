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
root_dir=/users/$USER/diezDecomp-main/tests
cd $root_dir

# ------------------------------ transpose ------------------------------
cd $root_dir/transpose
for trial_subfolder in $root_dir/transpose/trials/* ; do
  cd $trial_subfolder/input_transp
  nproc=$(ls *__info.dat | wc -l)
  cd $root_dir/transpose
  input_arg="$(basename -- $trial_subfolder)"
  echo $trial_subfolder $nproc $input_arg
  if [ -e "$trial_subfolder/use_gpu.txt" ]
  then
      echo "Running on GPU: ${trial_subfolder}"
      srun  -n $nproc ./main_transp_gpu $input_arg
  else
      echo "Running on CPU: ${trial_subfolder}"
      srun  -n $nproc ./main_transp_cpu $input_arg
  fi
  rm core
done

# ------------------------------ halo ------------------------------
cd $root_dir/halo
for trial_subfolder in $root_dir/halo/trials/* ; do
  cd $trial_subfolder/input_halo
  nproc=$(ls *__info.dat | wc -l)
  input_arg="$(basename -- $trial_subfolder)"
  echo $trial_subfolder $nproc $input_arg
  cd $root_dir/halo
  if [ -e "$trial_subfolder/use_gpu.txt" ]
  then
      echo "Running on GPU: ${trial_subfolder}"
      srun  -n $nproc ./main_halo_gpu $input_arg
  else
      echo "Running on CPU: ${trial_subfolder}"
      srun  -n $nproc ./main_halo_cpu $input_arg
  fi
  rm core
done

python3 $root_dir/transpose/error_checker.py
python3 $root_dir/halo/error_checker.py
