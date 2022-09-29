# LED_PVAE

<br/>
<p align="center"><img src="imgs/logo.png" width=500 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap?color=black)

The physics-informed variational autoencoder is a framework that uses self-supervised learning for reconstruction in sparse computational imaging. This repository `LED_PVAE` implements the physics-informed variational autoencoder for LED array microscopy, when the number of collected images is much less than the number of LEDs in the array.

# Overview
The figure below shows the overview of the end-to-end `LED_PVAE` pipeline.

<p align="center"><img src="imgs/intro_VAE_full.png" width=700 /></p>

The main algorithm comprising the `LED_PVAE` is inspired by the variational autoencoder. This repository allows creation of synthetic object datasets and generation of corresponding noisy intensity images. Object reconstruction from the intensity images can be performed with the physics-informed variational autoencoder or a standard gradient-based iterative algorithm. Options are included for using synthetic data or real experimental data collected on an LED array microscope. Code is included for visualization and comparison of results, and instructions are below for reproducing all the figures of the paper. An experimental dataset of frog blood smear images for use with this repository is available on [figshare](https://figshare.com/s/635499acfdcdf0893750).

# Table of contents
1. [Installation](#Installation)
2. [Synthetic Foam Dataset](#Foam)
3. [Synthetic 3-Dimensional MNIST Dataset](#MNIST)
4. [Experimental Dataset](#Experimental)

   
# Installation <a name="Installation"></a>

First create a `conda` environment:

```
conda env create -f environment.yml
conda activate LED
```

Once you're done with the above step, you need to use `pip install` to finish installing all the dependencies, using:

```
pip install -r requirements.txt
```

Finally, navigate to the folder where you want the repository and clone the repository:

```
git clone https://github.com/vganapati/LED_PVAE.git
```

And you're all set!


## Synthetic Foam Dataset

Navigate to the `LED_PVAE` directory.

First, create the dataset of objects:
```
python SyntheticMNIST_multislice.py --save_path foam_v2_pac1 --td --tn_train 10000 --tn_test 0 --rad 3 --Nx 128 --Ny 128 --dti 5 --rmf 0.9 -f 0 --ns 1 --pac 1
```

Next, create the multiplexed illumination patterns and emulate the corresponding intensity images (i.e. measurements) for different noise levels (to get the same illumination patterns for every noise level, do not run commands below in parallel, complete the first command before running the others):
```
python create_multiplexed.py --input_path dataset_foam_v2_pac1 --save_tag pnm1e4_dm01_p4 --pnm 1e4 --dm 0.1 -p 4

python create_multiplexed.py --input_path dataset_foam_v2_pac1 --save_tag pnm1e2_dm01_p4 --pnm 1e2 --dm 0.1 -p 4 --save_tag_alpha pnm1e4_dm01_p4

python create_multiplexed.py --input_path dataset_foam_v2_pac1 --save_tag pnm1e3_dm01_p4 --pnm 1e3 --dm 0.1 -p 4 --save_tag_alpha pnm1e4_dm01_p4

python create_multiplexed.py --input_path dataset_foam_v2_pac1 --save_tag pnm1e5_dm01_p4 --pnm 1e5 --dm 0.1 -p 4 --save_tag_alpha pnm1e4_dm01_p4
```

Next, do the same as above, except fixing all illumination patterns to be the **same:
```
python create_multiplexed.py --input_path dataset_foam_v2_pac1 --save_tag pnm1e4_single_dm01_p4 --pnm 1e4 --dm 0.1 -p 4 --single

python create_multiplexed.py --input_path dataset_foam_v2_pac1 --save_tag pnm1e2_single_dm01_p4 --pnm 1e2 --dm 0.1 -p 4 --single --save_tag_alpha pnm1e4_single_dm01_p4

python create_multiplexed.py --input_path dataset_foam_v2_pac1 --save_tag pnm1e3_single_dm01_p4 --pnm 1e3 --dm 0.1 -p 4 --single --save_tag_alpha pnm1e4_single_dm01_p4

python create_multiplexed.py --input_path dataset_foam_v2_pac1 --save_tag pnm1e5_single_dm01_p4 --pnm 1e5 --dm 0.1 -p 4 --single --save_tag_alpha pnm1e4_single_dm01_p4
```

Run the following to train the P-VAE on the dataset with different illumination patterns for every object:
```
./scripts/sweep.sh dataset_foam_v2_pac1 foam_pac1 false
```

Run the following to train the P-VAE on the dataset with the **same illumination patterns for every object:
```
./scripts/sweep.sh dataset_foam_v2_pac1 foam_pac1 true
```

STOPPED HERE

Run the following to train the P-VAE on a training dataset of size 1 (object index 0 of the complete dataset, using the illumination pattern from the different illuminations):
```
./scripts/sweep_single_pattern.sh dataset_foam_v2_pac1 foam_pac1
```

Run the following to train the P-VAE on a training dataset of size 1 (object index 0 of the complete dataset, using the illumination pattern from the case where they are all the same):
```
./scripts/sweep_single_pattern.sh dataset_foam_v2_pac1 foam_pac1
```


Run the following for standard iterative optimization:
```
./optimizer_sweep.sh dataset_foam_v2_pac1 10000 5 1e-3
```

Analyze/visualize the data


Make all plots from the paper



## Synthetic 3D MNIST Dataset

First, create the dataset:

```
python SyntheticMNIST_multislice.py --save_path MNIST_multislice_v2_test --td --tn_train 100 --tn_test 5 --rad 3 --Nx 32 --Ny 32 --dti 0 --rmf 0.9 --ns 2 -f -5 
```

Next, create the multiplexed illumination patterns and emulate the corresponding intensity images for different noise levels (to get the same illumination patterns for every noise level, do not run commands below in parallel):

```
first noise level: DONE
bsub -oo output_single -J job3 -n 4 -gpu "num=1" -q gpu_any python create_multiplexed.py --input_path dataset_MNIST_multislice_v2 --save_tag pnm1e4_single_dm01_p4 --pnm 1e4 --dm 0.1 -p 4 --single

DONE
other noise levels:

bsub -oo output_single -J job3 -n 4 -gpu "num=1" -q gpu_any python create_multiplexed.py --input_path dataset_MNIST_multislice_v2 --save_tag pnm1e2_single_dm01_p4 --pnm 1e2 --dm 0.1 -p 4 --single --save_tag_alpha pnm1e4_single_dm01_p4

bsub -oo output_single -J job3 -n 4 -gpu "num=1" -q gpu_any python create_multiplexed.py --input_path dataset_MNIST_multislice_v2 --save_tag pnm1e3_single_dm01_p4 --pnm 1e3 --dm 0.1 -p 4 --single --save_tag_alpha pnm1e4_single_dm01_p4

bsub -oo output_single -J job3 -n 4 -gpu "num=1" -q gpu_any python create_multiplexed.py --input_path dataset_MNIST_multislice_v2 --save_tag pnm1e5_single_dm01_p4 --pnm 1e5 --dm 0.1 -p 4 --single --save_tag_alpha pnm1e4_single_dm01_p4

```

RUN ME
./optimizer_sweep_single.sh dataset_MNIST_multislice_v2 10000 5 1e-3

RUN ME
./sweep.sh dataset_MNIST_multislice_v2 mnist_single (uncomment the save_tag_mult for the single pattern)
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single (uncomment the save_tag_mult for the single pattern)

Run sweep_single_example for objects 0-9: 
(uncomment the save_tag_mult for the same pattern for all objects)
SAVE_TAG_MULT single uncommented

./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 0
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 1
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 2
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 3
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 4
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 5
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 6
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 7
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 8
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist_single 9

SAVE_TAG_MULT single commented

./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 0
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 1
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 2
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 3
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 4
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 5
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 6
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 7
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 8
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist 9


Download
dataset_MNIST_multislice_v2/pnm1e3_dm01_p4

scp -r ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/AdaptiveFourierML/dataset_MNIST_multislice_v2/pnm1e3_dm01_p4 /Users/vganapa1/Dropbox/Github/AdaptiveFourierML/dataset_MNIST_multislice_v2


Optimizer training runs:
python optimizer_fpm_multislice_v2.py -i 2000 -b 1 -p 5 --t2 1e-3 --input_data dataset_MNIST_multislice_v2_test --obj_name training/example_000000 --find_ground_truth



----
Synthetic Data Pipeline:
SyntheticMNIST_multislice.py
create_multiplexed.py
./sweep.sh $INPUT_PATH_VEC $SAVE_TAG
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist (can be run at the same time as the above sweep, runs for 1 training example)
optimizer_sweep.sh (can be run at the same time as the above sweeps)
download_optimizer_sweep.sh
final_analyze_table.py
final_analyze_lines.py
plotter_helper_all_lowres.py
----

## Experimental Dataset


Commands for real data:
python compare_iterative.py --save_tag_recons _single_LED_iterative_short --save_path FPM_VAE_v2_ex_120_patt_4_Dirichlet


::Iterative single LED stack::
bsub -oo output_single_LED_iterative_short -J job0 -n 4 -gpu "num=1" -q gpu_v100 python merge_patches.py --save_tag_recons _single_LED_iterative_short  --alr 1e-3 -i 10000
python merge_patches_downloader.py --save_tag_recons _single_LED_iterative_short
visualize: python merge_patches_visualizer.py --iterative --save_tag_recons _single_LED_iterative_short

::Iterative multiplexed LED stack::
bsub -oo output_p1_LED_iterative_short -J job0 -n 4 -gpu "num=1" -q gpu_v100 python merge_patches.py --save_tag_recons _p1_LED_iterative_short --alr 1e-4 -i 10000 -p 1 --md _Dirichlet --use_mult
python merge_patches_downloader.py --save_tag_recons _p1_LED_iterative_short
visualize: --iterative --save_tag_recons _p1_LED_iterative_short --md _Dirichlet

::P-VAE::
bsub -oo output_frog_mult6_v3_100k -J job4 -n 8 -gpu "num=1" -q gpu_a100 python FPM_VAE_v3.py --input_path dataset_frog_blood_v3 --save_path frog_mult6_v3_100k -i 100000 -p 1 --td 87 --nb 3 --nfm 5 --nfmm 1.2 --dp 0 --lr 1e-4 --norm 100 --se 4 --ks 4 --il 3 --pnm 26869.35 -b 4 --klm 1.0 --klaf 1 --normal --visualize --en 0 --save_tag_mult real_multiplexed --real_data --uf 1 --use_window --real_mult --xcrop 512 --ycrop 512 --si 20000 --md _Dirichlet --train --fff_reconstruct


Creating all figures:



======================
Physics-informed variational autoencoder for LED array microscopy.

conda environment (make into yaml file):
conda create -n FPM3
conda activate FPM3
conda install -c conda-forge tensorflow
conda install tensorflow-probability
pip install imageio
pip install scikit-image
pip install tensorflow_datasets


Run the following commands:

datasets used:
dataset_MNIST_multislice_v2: --save_path MNIST_multislice_v2 --td --tn_train 100 --tn_test 100 --rad 3 --Nx 32 --Ny 32 --dti 0 --rmf 0.9 --ns 2 -f -5 
dataset_foam_v2_pac1 --save_path foam_v2_pac1 --td --tn_train 15000 --tn_test 10 --rad 3 --Nx 128 --Ny 128 --dti 5 --rmf 0.9 -f 0 --ns 1 --pac 1
128 



SyntheticMNIST_multislice.py
create_multiplexed.py
./sweep.sh $INPUT_PATH_VEC $SAVE_TAG
./sweep_single_example.sh dataset_MNIST_multislice_v2 mnist (can be run at the same time as the above sweep, runs for 1 training example)
optimizer_sweep.sh (can be run at the same time as the above sweeps)
download_optimizer_sweep.sh
final_analyze_table.py
final_analyze_lines.py
plotter_helper_all_lowres.py

optimizer_fpm_multislice_v3.py [-i 2000 -b 5 --t2 0 --input_data dataset_foam_v2 --save_tag pnm2e4_dm01_p4 --obj_ind 0 --pnm 26869.35 --mult -p 4]
FPM_VAE_v3.py --> download_opt.sh
optimizer_sweep_single.sh (for doing the iterative opts for the single pattern option, i.e. one pattern for all examples)
final_visualize_v2.py


=================================
=================================
CT Documentation below
=================================
=================================

<br/>
<p align="center"><img src="figures/logo.png" width=500 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap?color=black)

`CT_VAE` is a framework that uses self-supervised learning for computed tomography reconstruction from sparse sinograms (projection measurements). The framework is probabilisitic, inferring a prior distribution from sparse sinograms on a dataset of objects and calculating the posterior for the object reconstruction on each sinogram.

# Overview
The figure below shows the overview of the end-to-end `CT-VAE` pipeline.

<p align="center"><img src="figures/full_vae.png" width=700 /></p>

The main algorithm comprising the `CT-VAE` is inspired by the variational autoencoder. This repository allows creation of synthetic object datasets and generation of corresponding noisy, sparse sinograms. Object reconstruction from the sparse sinograms is performed with the physics-informed variational autoencoder. Code is included for visualization and comparison of results.  

# Table of contents
1. [Installation](#Installation)
2. [Running End-to-End](#Running)
3. [Reconstruction Algorithm Options](#Options)
4. [Reproducing Paper Figures](#PaperFigures)
   1. [Creating Datasets](#Datasets)
   2. [Reproducing Toy Dataset Results](#ToyDataset)
   3. [Reproducing Foam Dataset Results](#FoamDataset)
   
# Installation <a name="Installation"></a>

First create a `conda` environment:

```
conda env create -f environment.yml
conda activate CT
```

Once you're done with the above step, you need to use `pip install` to finish installing all the dependencies, using:

```
pip install --upgrade -r requirements_upgrade.txt
pip install -r requirements.txt
pip install -U hpo-uq==1.3.14
```

Finally, navigate to the folder where you want the repository and clone the repository:

```
git clone https://github.com/vganapati/CT_VAE.git
```

And you're all set!

# Running End-to-End <a name="Running"></a>

These are instructions to create a small synthetic dataset and quickly check that the code is working. Due to the fast training times and small size of the dataset, reconstruction results are expected to be poor. See [Reproducing Paper Figures](#PaperFigures) for instructions on creating larger datasets and using longer training times.

Activate your environment if not already activated:

```
conda activate CT
```

Navigate to the CT_VAE directory. 
Next, set your PYTHONPATH to include the current directory:

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Run the following to create a synthetic foam dataset of 50 examples, saved in the subfolder `dataset_foam` of the current directory:

```
python scripts/create_foam_images.py -n 50
python scripts/images_to_sinograms.py -n 50
```

Finally, train the physics-informed variational autoencoder; results are saved in subfolder `ct_vae_test` of the current directory.

```
python bin/main_ct_vae.py --input_path dataset_foam --save_path ct_vae_test -b 5 --pnm 1e4 -i 1000 --td 50 --normal --visualize --nsa 20 --ns 2 --api 20 --pnm_start 1e3 --random --algorithms gridrec --train

```

# Reconstruction Algorithm Options <a name="Options"></a>

There are several options for running the reconstruction algorithm with `python bin/main_ct_vae.py`.

```
usage: main_ct_vae.py [-h] [--ae ADAM_EPSILON] [-b BATCH_SIZE] [--ns NUM_SAMPLES] [--det] [--dp DROPOUT_PROB]
                      [--en EXAMPLE_NUM] [-i NUM_ITER] [--ik INTERMEDIATE_KERNEL] [--il INTERMEDIATE_LAYERS]
                      [--input_path INPUT_PATH] [--klaf KL_ANNEAL_FACTOR] [--klm KL_MULTIPLIER] [--ks KERNEL_SIZE]
                      [--lr LEARNING_RATE] [--nb NUM_BLOCKS] [--nfm NUM_FEATURE_MAPS]
                      [--nfmm NUM_FEATURE_MAPS_MULTIPLIER] [--norm NORM] [--normal] [--nsa NUM_SPARSE_ANGLES]
                      [--api ANGLES_PER_ITER] [--pnm POISSON_NOISE_MULTIPLIER] [--pnm_start PNM_START]
                      [--train_pnm] [-r RESTORE_NUM] [--random] [--restore] [--save_path SAVE_PATH]
                      [--se STRIDE_ENCODE] [--si SAVE_INTERVAL] [--td TRUNCATE_DATASET] [--train] [--ufs] [--ulc]
                      [--visualize] [--pixel_dist] [--real] [--no_pad] [--toy_masks]
                      [--algorithms ALGORITHMS [ALGORITHMS ...]] [--no_final_eval]

Get command line args

optional arguments:
  -h, --help            show this help message and exit
  --ae ADAM_EPSILON     adam_epsilon
  -b BATCH_SIZE         batch size
  --ns NUM_SAMPLES      number of times to sample VAE in training
  --det                 no latent variable, simply maximizes log probability of output_dist
  --dp DROPOUT_PROB     dropout_prob, percentage of nodes that are dropped
  --en EXAMPLE_NUM      example index for visualization
  -i NUM_ITER           number of training iterations
  --ik INTERMEDIATE_KERNEL
                        intermediate_kernel for model_encode
  --il INTERMEDIATE_LAYERS
                        intermediate_layers for model_encode
  --input_path INPUT_PATH
                        path to folder containing training data
  --klaf KL_ANNEAL_FACTOR
                        multiply kl_anneal by this factor each iteration
  --klm KL_MULTIPLIER   multiply the kl_divergence term in the loss function by this factor
  --ks KERNEL_SIZE      kernel size in model_encode_I_m
  --lr LEARNING_RATE    learning rate
  --nb NUM_BLOCKS       num convolution blocks in model_encode
  --nfm NUM_FEATURE_MAPS
                        number of features in the first block of model_encode
  --nfmm NUM_FEATURE_MAPS_MULTIPLIER
                        multiplier of features for each block of model_encode
  --norm NORM           gradient clipping by norm
  --normal              use a normal distribution as final distribution
  --nsa NUM_SPARSE_ANGLES
                        number of angles to image per sample (dose remains the same)
  --api ANGLES_PER_ITER
                        number of angles to check per iteration (stochastic optimization)
  --pnm POISSON_NOISE_MULTIPLIER
                        poisson noise multiplier, higher value means higher SNR
  --pnm_start PNM_START
                        poisson noise multiplier starting value, anneals to pnm value
  --train_pnm           if True, make poisson_noise_multiplier a trainable variable
  -r RESTORE_NUM        checkpoint number to restore from
  --random              if True, randomly pick angles for masks
  --restore             restore from previous training
  --save_path SAVE_PATH
                        path to save output
  --se STRIDE_ENCODE    convolution stride in model_encode_I_m
  --si SAVE_INTERVAL    save_interval for checkpoints and intermediate values
  --td TRUNCATE_DATASET
                        truncate_dataset by this value to not load in entire dataset; overriden when restoring a
                        net
  --train               run the training loop
  --ufs                 use the first skip connection in the unet
  --ulc                 uses latest checkpoint, overrides -r
  --visualize           visualize results
  --pixel_dist          get distribution of each pixel in final reconstruction
  --real                denotes real data, does not simulate noise
  --no_pad              sinograms have no zero-padding
  --toy_masks           uses the toy masks
  --algorithms ALGORITHMS [ALGORITHMS ...]
                        list of initial algorithms to use
  --no_final_eval       skips the final evaluation
```

# Reproducing Paper Figures <a name="PaperFigures"></a>

## Creating Datasets <a name="Datasets"></a>

Run the following once to create the datasets (both `dataset_toy` and `dataset_foam`):

Activate your environment if not already activated:

```
conda activate CT
```

Navigate to the CT_VAE directory. 
Next, set your PYTHONPATH to include the current directory:

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Run the following to create a synthetic foam dataset of 1,000 examples, saved in the subfolder `dataset_foam` of the current directory:

```
python scripts/create_foam_images.py -n 1000
python scripts/images_to_sinograms.py -n 1000
```

Run the following to create a synthetic toy dataset of 1,024 examples, saved in the subfolder `dataset_toy_discrete2` of the current directory:

```
- python scripts/create_toy_images.py -n 1024
- python scripts/images_to_sinograms.py -n 1024 --toy
```

## Reproducing Toy Dataset Results <a name="ToyDataset"></a>

Train the physics-informed variational autoencoder with the following options:

```
python bin/main_ct_vae.py --input_path dataset_toy_discrete2 --save_path toy_vae -b 4 --pnm 10000 -i 100000 --td 1024 --train --nsa 1 --ik 2 --il 5 --ks 2 --nb 3 --api 2 --se 1 --no_pad --ns 10 --pnm_start 1000 --si 100000 --normal --visualize --toy_masks

```

To create marginal posterior probability distributions for each pixel, results saved in the subfolder `toy_vae` of the current directory:

```
python bin/main_ct_vae.py --input_path dataset_toy_discrete2 --save_path toy_vae -b 4 --pnm 10000 -i 100000 --td 1024 --nsa 1 --ik 2 --il 5 --ks 2 --nb 3 --api 2 --se 1 --no_pad --ns 10 --pnm_start 1000 --si 100000 --normal --visualize --toy_masks --pixel_dist --restore --ulc --en 0

python bin/main_ct_vae.py --input_path dataset_toy_discrete2 --save_path toy_vae -b 4 --pnm 10000 -i 100000 --td 1024 --nsa 1 --ik 2 --il 5 --ks 2 --nb 3 --api 2 --se 1 --no_pad --ns 10 --pnm_start 1000 --si 100000 --normal --visualize --toy_masks --pixel_dist --restore --ulc --en 1

python bin/main_ct_vae.py --input_path dataset_toy_discrete2 --save_path toy_vae -b 4 --pnm 10000 -i 100000 --td 1024 --nsa 1 --ik 2 --il 5 --ks 2 --nb 3 --api 2 --se 1 --no_pad --ns 10 --pnm_start 1000 --si 100000 --normal --visualize --toy_masks --pixel_dist --restore --ulc --en 2

python bin/main_ct_vae.py --input_path dataset_toy_discrete2 --save_path toy_vae -b 4 --pnm 10000 -i 100000 --td 1024 --nsa 1 --ik 2 --il 5 --ks 2 --nb 3 --api 2 --se 1 --no_pad --ns 10 --pnm_start 1000 --si 100000 --normal --visualize --toy_masks --pixel_dist --restore --ulc --en 3
```


## Reproducing Foam Dataset Results <a name="FoamDataset"></a>

Train the physics-informed variational autoencoder with the following options, results saved in the subfolder `foam_vae` of the current directory:

```
python bin/main_ct_vae.py --input_path dataset_foam --save_path foam_vae -b 10 --pnm 10000 -i 100000 --td 1000 --normal --visualize --nsa 20 --ns 2 --api 20 --pnm_start 1000 --si 100000 --random --algorithms sirt tv fbp gridrec --train
```