#!/bin/bash
INPUT_PATH_VEC=$1
SAVE_TAG=$2
declare -i COUNTER=0
NUM_EXAMPLES=0
for NOISE_LEVEL in 2 3 4 5
do
for NUM_PATTERNS in 1 2 3 4
do
NUM_EXAMPLES_FULL=$((10**$NUM_EXAMPLES))
NOISE_LEVEL_FULL=$((10**$NOISE_LEVEL))
SAVE_PATH=$SAVE_TAG'_noise_'$NOISE_LEVEL'_ex_'$NUM_EXAMPLES'_p_'$NUM_PATTERNS
#SAVE_TAG_MULT='pnm1e'$NOISE_LEVEL'_dm01_p4'
SAVE_TAG_MULT='pnm1e'$NOISE_LEVEL'_single_dm01_p4' #single pattern
bsub -oo output1_$SAVE_TAG'_'$COUNTER -J job3 -n 4 -gpu "num=1" -q gpu_tesla python FPM_VAE_v3.py --input_path_vec $INPUT_PATH_VEC --save_path $SAVE_PATH -i 50000 -p $NUM_PATTERNS --ms 1 --td $NUM_EXAMPLES_FULL --nb 3 --nfm0 10 --nfmm0 1.5 --dp 0 --lr 1e-4 --norm 100 --se0 4 --ks0 4 --pi_iter --use_coords --use_bias --il 3 --pnm $NOISE_LEVEL_FULL -b $NUM_EXAMPLES_FULL --reconstruct --klm 1.0 --klaf 1 --ta --rand_i --sfv 0 --unsup --normal --cp --train --visualize --en 0 --sci -1 --ndlp 1 --sfd 0 --dm 0.1 --save_tag_mult $SAVE_TAG_MULT --final_train
COUNTER+=1
done
done
