#!/bin/bash
INPUT_DATA=$1
NUM_ITER=$2
BATCH_SIZE=$3
LEARNING_RATE=$4
SINGLE_PATT=$5
for NOISE_LEVEL in 2 3 4 5
do
if $SINGLE_PATT
then
SAVE_TAG_MULT='pnm1e'$NOISE_LEVEL'_single_dm01_p4'
else
SAVE_TAG_MULT='pnm1e'$NOISE_LEVEL'_dm01_p4'
fi
NOISE_LEVEL_FULL=$((10**$NOISE_LEVEL))
for OBJ_IND in 0 1 2 3 4 5 6 7 8 9
do
# single leds l2 reg
python optimizer_fpm_multislice_v3.py -i $NUM_ITER -b $BATCH_SIZE --t2 1e-2 --alr $LEARNING_RATE --input_data $INPUT_DATA --obj_ind $OBJ_IND --pnm $NOISE_LEVEL_FULL
# single leds l1 reg
python optimizer_fpm_multislice_v3.py -i $NUM_ITER -b $BATCH_SIZE --t2 1e-2 --alr $LEARNING_RATE --input_data $INPUT_DATA --obj_ind $OBJ_IND --pnm $NOISE_LEVEL_FULL --l1
# single leds no reg
python optimizer_fpm_multislice_v3.py -i $NUM_ITER -b $BATCH_SIZE --t2 0 --alr $LEARNING_RATE --input_data $INPUT_DATA --obj_ind $OBJ_IND --pnm $NOISE_LEVEL_FULL
for NUM_PATTERNS in 1 2 3 4
do
# multiplexed l2 reg
python optimizer_fpm_multislice_v3.py -i $NUM_ITER -b $BATCH_SIZE --t2 1e-2 --alr $LEARNING_RATE --input_data $INPUT_DATA --obj_ind $OBJ_IND --pnm $NOISE_LEVEL_FULL --mult -p $NUM_PATTERNS --save_tag $SAVE_TAG_MULT
# multiplexed l1 reg
python optimizer_fpm_multislice_v3.py -i $NUM_ITER -b $BATCH_SIZE --t2 1e-2 --alr $LEARNING_RATE --input_data $INPUT_DATA --obj_ind $OBJ_IND --pnm $NOISE_LEVEL_FULL --mult -p $NUM_PATTERNS --save_tag $SAVE_TAG_MULT --l1
# multiplexed no reg
python optimizer_fpm_multislice_v3.py -i $NUM_ITER -b $BATCH_SIZE --t2 0 --alr $LEARNING_RATE --input_data $INPUT_DATA --obj_ind $OBJ_IND --pnm $NOISE_LEVEL_FULL --mult -p $NUM_PATTERNS --save_tag $SAVE_TAG_MULT
done
done
done







