#!/bin/bash

# Change the absolute path first!
DATA_ROOT_DIR="/home/ubuntu/ahmed-etri"

OUTPUT_PPO_DIR="output_ppo_training"
EPOCHS=100

# Checkpoints (shared across all calls — overwritten each episode)
STATE_ENCODER_CHECKPOINT="checkpoints/state_encoder.pth"
PPO_POLICY_CHECKPOINT="checkpoints/ppo_policy.pth"

# Shared log directory — ppo_losses.csv accumulates here across all episodes
LOG_DIR="${OUTPUT_PPO_DIR}/logs"

SCENES=(
    tanks_templates:horse
    # tanks_templates:ballroom
    tanks_templates:barn
    # tanks_templates:church
    tanks_templates:family
    # tanks_templates:francis
    tanks_templates:ignatius
    # tanks_templates:museum
    MVimgNet:bench
    # MVimgNet:bicycle
    MVimgNet:car
    # MVimgNet:chair
    MVimgNet:ladder
    # MVimgNet:suv
    MVimgNet:table
)

N_VIEWS=(
    3
    6
    12
)

GS_TRAIN_ITERS=(
    1000
    # 2000
)

N_SCENES=${#SCENES[@]}

echo "=========================================="
echo "PPO Policy Training"
echo "Total Epochs  : ${EPOCHS}"
echo "Scenes        : ${SCENES[*]}"
echo "N-Views       : ${N_VIEWS[*]}"
echo "GS Iters      : ${GS_TRAIN_ITERS[*]}"
echo "Log Dir       : ${LOG_DIR}"
echo "=========================================="

mkdir -p "${LOG_DIR}"

# Global episode counter — increments once per (epoch × n_view × gs_iter × scene)
EPISODE=1

for EPOCH in $(seq 1 ${EPOCHS}); do
    echo ""
    echo "=========================================="
    echo "EPOCH ${EPOCH}/${EPOCHS}"
    echo "=========================================="

    for N_VIEW in "${N_VIEWS[@]}"; do
        for gs_train_iter in "${GS_TRAIN_ITERS[@]}"; do
            for DATASET_SCENE in "${SCENES[@]}"; do
                DATASET="${DATASET_SCENE%%:*}"
                SCENE="${DATASET_SCENE#*:}"

                SOURCE_PATH="${DATA_ROOT_DIR}/${DATASET}/${SCENE}/24_views/"
                MODEL_PATH="${OUTPUT_PPO_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views/epoch_${EPOCH}"

                mkdir -p "${MODEL_PATH}"

                echo ""
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Episode ${EPISODE} — Epoch ${EPOCH}/${EPOCHS}  scene: ${DATASET}/${SCENE}  n_views=${N_VIEW}  iters=${gs_train_iter}"

                # ── (1) Co-visible Global Geometry Initialisation ──────────────
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] init_geo ..."
                CUDA_VISIBLE_DEVICES=0 python -W ignore ./init_geo.py \
                    -s "${SOURCE_PATH}" \
                    -m "${MODEL_PATH}" \
                    --n_views "${N_VIEW}" \
                    --focal_avg \
                    --co_vis_dsp \
                    --conf_aware_ranking \
                    > "${MODEL_PATH}/01_init_geo.log" 2>&1
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] init_geo done. Log: ${MODEL_PATH}/01_init_geo.log"

                # ── (2) PPO single-episode training ────────────────────────────
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] train_ppo (episode ${EPISODE}) ..."
                CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ./train_ppo.py \
                    -s "${SOURCE_PATH}" \
                    -m "${MODEL_PATH}" \
                    -r 1 \
                    --n_views "${N_VIEW}" \
                    --iterations "${gs_train_iter}" \
                    --pp_optimizer \
                    --optim_pose \
                    --policy_backbone transformer \
                    --train_state_encoder \
                    --load_state_encoder "${STATE_ENCODER_CHECKPOINT}" \
                    --save_state_encoder "${STATE_ENCODER_CHECKPOINT}" \
                    --load_ppo_policy "${PPO_POLICY_CHECKPOINT}" \
                    --save_ppo_policy "${PPO_POLICY_CHECKPOINT}" \
                    --log_dir "${LOG_DIR}" \
                    --epoch "${EPOCH}" \
                    --episode "${EPISODE}" \
                    > "${MODEL_PATH}/02_train_ppo.log" 2>&1
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] train_ppo done. Log: ${MODEL_PATH}/02_train_ppo.log"

                EPISODE=$(( EPISODE + 1 ))
            done
        done
    done
done

# ── Phase 2: Plot losses ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Plotting PPO Losses"
echo "=========================================="

PLOTS_DIR="${OUTPUT_PPO_DIR}/plots"
mkdir -p "${PLOTS_DIR}"

python plot_ppo_losses.py \
    --output_dir "${LOG_DIR}" \
    --epochs "${EPOCHS}" \
    --plots_dir "${PLOTS_DIR}" \
    --scenes "${SCENES[@]}"

echo ""
echo "=========================================="
echo "All PPO Training Complete!"
echo "Logs    : ${LOG_DIR}"
echo "Plots   : ${PLOTS_DIR}"
echo "=========================================="
