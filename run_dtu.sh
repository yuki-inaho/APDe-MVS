#!/bin/bash
# make 
cmake --build ./build --target APD -j 4

##############################
# DTU test depth
##############################
python run.py \
   --APD_path $HOME/work/APD-explore/build/APD \
   --data_dir $HOME/data/DTU/test \
   --memory_cache \
   --no_fuse \
   --work_num 2 \
   --gpu_num 4 \
   --backup_code
##############################
# DTU test fuse
##############################
python run.py \
   --APD_path $HOME/work/APD-explore/build/APD \
   --data_dir $HOME/data/DTU/test \
   --only_fuse \
   --work_num 22
##############################
# scan run
##############################
# python run.py \
#    --APD_path $HOME/work/APD-explore/build/APD \
#    --data_dir $HOME/data/DTU/test \
#    --memory_cache \
#    --no_fuse \
#    --backup_code \
#    --scans scan77 
# python run.py \
#    --APD_path $HOME/work/APD-explore/build/APD \
#    --data_dir $HOME/data/DTU/test \
#    --only_fuse \
#    --scans scan77 

email -c "Hello zzj, the train task <APD-explore/run_dtu.sh> with jobid $SLURM_JOB_ID is done"
scancel $SLURM_JOB_ID