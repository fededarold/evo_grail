#!/bin/bash
#export OMP_NUM_THREADS=1
# export OMP_SCHEDULE=STATIC
# export OMP_PROC_BIND=CLOSE
python3 mp_test_grail_beta_v2.py 
# k=1
# for i in {10..14}
# do
        # echo "conf $i"
        # echo $( expr $i + $k )
        # taskset -cp $i
        # taskset -c $i python3 run_experiment_dqn_pong.py --conf $( expr $i + $k ) &
# done
# wait
