#!/bin/bash
# make 
cmake --build ./build --target APD -j 4

##############################
# ETH3D all depth
##############################
python run.py \
  --APD_path ./build/APD \
  --data_dir $HOME/data/ETH3D/data \
  --memory_cache \
  --no_fuse \
  --gpu_num 3 \
  --backup_code
##############################
# ETH3D all fuse
##############################
python run.py \
    --APD_path ./build/APD \
    --data_dir $HOME/data/ETH3D/data \
    --only_fuse \
    --work_num 13 \
    --backup_code
##############################
# ETH3D train depth
##############################
# python run.py \
#   --APD_path ./build/APD \
#   --data_dir $HOME/data/ETH3D/data \
#   --memory_cache \
#   --no_fuse \
#   --gpu_num 4 \
#   --backup_code \
#   --ETH3D_train
##############################
# ETH3D train fuse
##############################
# python run.py \
#   --APD_path ./build/APD \
#   --data_dir $HOME/data/ETH3D/data \
#   --only_fuse \
#   --work_num 6 \
#   --ETH3D_train
##############################
# ETH3D test depth
##############################
# python run.py \
#     --APD_path ./build/APD \
#     --data_dir $HOME/data/ETH3D/data \
#     --memory_cache \
#     --no_fuse \
#     --gpu_num 4 \
#     --backup_code \
#     --ETH3D_test
##############################
# ETH3D test fuse
##############################
# python run.py \
#     --APD_path ./build/APD \
#     --data_dir $HOME/data/ETH3D/data \
#     --only_fuse \
#     --work_num 6 \
#     --backup_code \
#     --ETH3D_test
##############################
# scan run
##############################
# python run.py \
#     --APD_path ./build/APD \
#     --data_dir $HOME/data/ETH3D/data \
#     --memory_cache \
#     --gpu_num 2 \
#     --no_fuse \
#     --backup_code \
#     --scans office pipes \
#     --export_anchor
# python run.py \
#     --APD_path ./build/APD \
#     --data_dir $HOME/data/ETH3D/data \
#     --work_num 2 \
#     --only_fuse \
#     --scans office pipes

email -c "Hello zzj, the train task <APD-explore/run_eth.sh> with jobid $SLURM_JOB_ID is done"
scancel $SLURM_JOB_ID