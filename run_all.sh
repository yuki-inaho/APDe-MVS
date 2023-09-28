#!/bin/bash
# make 
cmake --build ./build --target APD -j 4

##############################
# depth
##############################
# DTU
python run.py \
   --APD_path ./build/APD \
   --data_dir $HOME/data/DTU/test \
   --memory_cache \
   --no_fuse \
   --gpu_num 4 \
   --backup_code
# TaT
python run.py \
  --APD_path ./build/APD \
  --data_dir $HOME/data/TaT/data \
  --memory_cache \
  --no_fuse \
  --gpu_num 4 \
  --backup_code
# ETH3D
python run.py \
  --APD_path ./build/APD \
  --data_dir $HOME/data/ETH3D/data \
  --memory_cache \
  --no_fuse \
  --gpu_num 4 \
  --backup_code
##############################
# fuse
##############################
# # DTU
# python run.py \
#    --APD_path ./build/APD \
#    --data_dir $HOME/data/DTU/test \
#    --only_fuse \
#    --work_num 22
# # TaT
# python run.py \
#     --APD_path ./build/APD \
#     --data_dir $HOME/data/ETH3D/data \
#     --only_fuse \
#     --work_num 13 \
#     --backup_code
# # ETH3D
# python run.py \
#     --APD_path ./build/APD \
#     --data_dir $HOME/data/ETH3D/data \
#     --only_fuse \
#     --work_num 6 \
#     --backup_code

email -c "Hello zzj, the train task <APDe-MVS/run_all.sh> with jobid $SLURM_JOB_ID is done"
scancel $SLURM_JOB_ID