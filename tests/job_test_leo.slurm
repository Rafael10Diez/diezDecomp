#!/bin/bash
#SBATCH -A [please_write_project_code]
#SBATCH -N 2
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --cpus-per-task=1
#SBATCH --time 00:29:59     # format: HH:MM:SS
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --exclusive
#SBATCH --job-name=test

echo "Starting job $SLURM_JOB_ID at `date`"
echo "USER: ${USER}"

# ------------------------------- working directory -------------------------------
# please adjust working directory
root_dir=/leonardo/home/userexternal/$USER/diezDecomp-main/tests

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
