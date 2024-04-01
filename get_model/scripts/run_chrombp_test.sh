#!/bin/bash
#SBATCH --job-name=get_finetune
#SBATCH --output=/pmglocal/ty2514/%x_%j.out                # Output file
#SBATCH --error=/pmglocal/ty2514/%x_%j.err                 # Error file
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --account pmg
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G


# Set the path to save checkpoints
DATE=`date +%Y%m%d`
OUTPUT_DIR="/pmglocal/ty2514/Expression_Finetune_monocyte.Chr4&14.conv50.atac_loss.chrombpnet.shift10.R100L1000.augmented.${DATE}/"
# path to expression set
DATA_PATH='/pmglocal/ty2514/GET_STARTED/get_data'
PORT=7959

# path to expression set
DATA_PATH='/pmglocal/ty2514/GET_STARTED/get_data'
# path to environment
ROOT_DIR = '/pmglocal/ty2514/GET_STARTED'
PORT=7958
export NCCL_P2P_LEVEL=NVL

# Load mamba module
module load mamba
# Activate mamba environment
source activate ${ROOT_DIR}/mambaforge/get_started
# cd to right execution (codebase) directory
cd ${ROOT_DIR}/get_model

#"/burg/pmg/users/xf2217/get_data/checkpoint-399.from_shentong.R200L500.pth" \

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:$PORT get_model/finetune.py \
    --data_set "Expression_Finetune_monocyte.Chr4&14" \
    --eval_data_set "Expression_Finetune_monocyte.Chr4&14.Eval" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif_chrombpnet \
    --batch_size 16 \
    --num_workers 32 \
    --n_peaks_lower_bound 2 \
    --n_peaks_upper_bound 10 \
    --center_expand_target 1000 \
    --preload_count 200 \
    --random_shift_peak \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --save_ckpt_freq 1 \
    --n_packs 1 \
    --lr 1e-4 \
    --opt adamw \
    --wandb_project_name "chrombpnet" \
    --wandb_run_name "Expression_Finetune_K562.Chr4&14.conv50.atac_loss.nofreeze.nodepth.gap50.shift10.R100L1000.augmented."${DATE} \
    --eval_freq 2 \
    --dist_eval \
    --eval_tss \
    --leave_out_celltypes "Mono" \
    --leave_out_chromosomes "chr4,chr14" \
    --criterion "mse" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 20 \
    --epochs 100 \
    --num_region_per_sample 10 \
    --output_dir ${OUTPUT_DIR}
