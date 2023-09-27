#!/bin/bash
# make 
cmake --build ./build --target APD -j 4

##############################
# TaT all depth
##############################
python run.py \
  --APD_path $HOME/work/APD-explore/build/APD \
  --data_dir $HOME/data/TaT/data \
  --memory_cache \
  --no_fuse \
  --gpu_num 3 \
  --backup_code
##############################
# TaT all fuse
##############################
python run.py \
  --APD_path $HOME/work/APD-explore/build/APD \
  --data_dir $HOME/data/TaT/data \
  --only_fuse \
  --work_num 4 \
  --backup_code
##############################
# TaT intermediate depth
##############################
# python run.py \
#     --APD_path $HOME/work/APD-explore/build/APD \
#     --data_dir $HOME/data/TaT/data \
#     --memory_cache \
#     --no_fuse \
#     --gpu_num 3 \
#     --backup_code \
#     --TaT_intermediate
##############################
# TaT intermediate fuse
##############################
# python run.py \
#     --APD_path $HOME/work/APD-explore/build/APD \
#     --data_dir $HOME/data/TaT/data \
#     --only_fuse \
#     --work_num 4 \
#     --TaT_intermediate
##############################
# TaT advanced depth
##############################
# python run.py \
#     --APD_path $HOME/work/APD-explore/build/APD \
#     --data_dir $HOME/data/TaT/data \
#     --memory_cache \
#     --no_fuse \
#     --gpu_num 3 \
#     --backup_code \
#     --TaT_advanced
##############################
# TaT advanced fuse
##############################
# python run.py \
#     --APD_path $HOME/work/APD-explore/build/APD \
#     --data_dir $HOME/data/TaT/data \
#     --only_fuse \
#     --work_num 4 \
#     --TaT_advanced
##############################
# scan run
##############################
# python run.py \
#     --APD_path $HOME/work/APD-explore/build/APD \
#     --data_dir $HOME/data/TaT/data \
#     --memory_cache \
#     --gpu_num 1 \
#     --backup_code \
#     --nofuse \
#     --scans Horse
# python run.py \
#    --APD_path $HOME/work/APD-explore/build/APD \
#    --data_dir $HOME/data/TaT/data \
#    --work_num 1 \
#    --only_fuse \
#    --scans Horse

email -c "Hello zzj, the train task <APD-explore/run_tat.sh> with jobid $SLURM_JOB_ID is done"
scancel $SLURM_JOB_ID