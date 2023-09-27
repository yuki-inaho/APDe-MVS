#!/bin/bash
# make 
cmake --build ./build --target APD -j 4

##############################
# scan run
##############################
python run.py \
   --APD_path $HOME/work/APD-explore/build/APD \
   --data_dir $HOME/data/temp \
   --memory_cache \
   --backup_code \
   --no_fuse \
   --scans output
python run.py \
  --APD_path $HOME/work/APD-explore/build/APD \
  --data_dir $HOME/data/temp \
  --only_fuse \
  --work_num 1 \
  --backup_code

email -c "Hello zzj, the train task <APD-explore/run_general.sh> with jobid $SLURM_JOB_ID is done"
scancel $SLURM_JOB_ID