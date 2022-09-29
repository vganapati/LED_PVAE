#!/bin/bash
INPUT_PATH=$1
SAVE_TAG=$2
SINGLE_PATT=$3
declare -i COUNTER=0
for NOISE_LEVEL in 2 3 4 5
do
for NUM_EXAMPLES in 1 2 3 4
do
for NUM_PATTERNS in 1 2 3 4
do
NUM_EXAMPLES_FULL=$((10**$NUM_EXAMPLES))
NOISE_LEVEL_FULL=$((10**$NOISE_LEVEL))
SAVE_PATH=$SAVE_TAG'_noise_'$NOISE_LEVEL'_ex_'$NUM_EXAMPLES'_p_'$NUM_PATTERNS
if $SINGLE_PATT
then
SAVE_TAG_MULT='pnm1e'$NOISE_LEVEL'_single_dm01_p4' #single pattern
else
SAVE_TAG_MULT='pnm1e'$NOISE_LEVEL'_dm01_p4'
fi
python FPM_VAE_v3.py --input_path $INPUT_PATH --save_path $SAVE_PATH -i 50000 -p $NUM_PATTERNS --td $NUM_EXAMPLES_FULL --nb 3 --nfm 10 --nfmm 1.5 --dp 0 --lr 1e-4 --norm 100 --se 4 --ks 4 --il 3 --pnm $NOISE_LEVEL_FULL -b 10 --klm 1.0 --klaf 1 --normal --train --visualize --en 0 --save_tag_mult $SAVE_TAG_MULT --final_train --vary_pupil
echo output_$SAVE_TAG'_'$COUNTER
COUNTER+=1
done
done
done