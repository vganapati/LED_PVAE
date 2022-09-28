# LED_PVAE

Commands for synthetic data:


Commands for real data:
bsub -oo output_frog_mult6_v3_100k -J job4 -n 8 -gpu "num=1" -q gpu_a100 python FPM_VAE_v3.py --input_path dataset_frog_blood_v3 --save_path frog_mult6_v3_100k -i 100000 -p 1 --td 87 --nb 3 --nfm 5 --nfmm 1.2 --dp 0 --lr 1e-4 --norm 100 --se 4 --ks 4 --il 3 --pnm 26869.35 -b 4 --klm 1.0 --klaf 1 --normal --visualize --en 0 --save_tag_mult real_multiplexed --real_data --uf 1 --use_window --real_mult --xcrop 512 --ycrop 512 --si 20000 --md _Dirichlet --train

bsub -oo output_frog_mult6_v3_100k_fff -J job4 -n 8 -gpu "num=1" -q gpu_a100 python FPM_VAE_v3.py --input_path dataset_frog_blood_v3 --save_path frog_mult6_v3_100k -i 100000 -p 1 --td 87 --nb 3 --nfm 5 --nfmm 1.2 --dp 0 --lr 1e-4 --norm 100 --se 4 --ks 4 --il 3 --pnm 26869.35 -b 4 --klm 1.0 --klaf 1 --normal --restore --ulc --visualize --en 0 --save_tag_mult real_multiplexed --real_data --uf 1 --use_window --real_mult --xcrop 512 --ycrop 512 --si 20000 --md _Dirichlet --fff_reconstruct





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