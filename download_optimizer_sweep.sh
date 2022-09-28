#!/bin/bash
INPUT_PATH='dataset_foam_2s_vary_pupil3'
SAVE_TAG='foam_vary_pupil'
GITHUB='FPM_VAE' # 'FPM_VAE' or 'AdaptiveFourierML'
IND_BATCHED=0
mkdir $INPUT_PATH
mkdir $INPUT_PATH'/training'
for EXAMPLE_NAME in 'training/example_000000' 'training/example_000001' 'training/example_000002' 'training/example_000003' 'training/example_000004' 'training/example_000005' 'training/example_000006' 'training/example_000007' 'training/example_000008' 'training/example_000009'
do
#Download iterative results
echo rm -r /Users/vganapa1/Dropbox/Github/$GITHUB/$INPUT_PATH/$EXAMPLE_NAME
echo scp -r ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/$GITHUB/$INPUT_PATH/$EXAMPLE_NAME /Users/vganapa1/Dropbox/Github/$GITHUB/$INPUT_PATH/$EXAMPLE_NAME
done
#Download neural network results
for NOISE_LEVEL in 2 3 4 5
do
for NUM_EXAMPLES in 1 2 3 4 #0 1 2 3 4
do
for NUM_PATTERNS in 1 2 3 4
do
#for EXAMPLE_NUM in 0 1 2 3 4 5 6 7 8 9
#do
NUM_EXAMPLES_FULL=$((10**$NUM_EXAMPLES))
NOISE_LEVEL_FULL=$((10**$NOISE_LEVEL))
SAVE_PATH=$SAVE_TAG'_noise_'$NOISE_LEVEL'_ex_'$NUM_EXAMPLES'_p_'$NUM_PATTERNS 
#SAVE_PATH=$SAVE_TAG'_noise_'$NOISE_LEVEL'_ex_'$NUM_EXAMPLES'_p_'$NUM_PATTERNS'_ex'$EXAMPLE_NUM
echo $SAVE_PATH
mkdir /Users/vganapa1/Dropbox/Github/$GITHUB/$SAVE_PATH
scp ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/$GITHUB/$SAVE_PATH/training/all_filtered_obj$IND_BATCHED.npy /Users/vganapa1/Dropbox/Github/$GITHUB/$SAVE_PATH
#scp ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/$GITHUB/$SAVE_PATH/training/entropy_vec$IND_BATCHED.npy /Users/vganapa1/Dropbox/Github/$GITHUB/$SAVE_PATH
#done
done
done
done