#!/bin/bash

# Change the absolute path first!
DATA_ROOT_DIR="/home/ubuntu/ahmed-etri"

OUTPUT_PPO_DIR="output_ppo_training"
EPOCHS=100
gs_train_iter=1000

# Checkpoints
STATE_ENCODER_CHECKPOINT="checkpoints/state_encoder_final.pth"
PPO_POLICY_CHECKPOINT="checkpoints/ppo_policy.pth"

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

echo "=========================================="
echo "PPO Policy Multi-Scene Training"
echo "Total Epochs  : ${EPOCHS}"
echo "Train Iters   : ${gs_train_iter}"
echo "Scenes        : ${SCENES[@]}"
echo "N-Views       : ${N_VIEWS[@]}"
echo "=========================================="

# ── Phase 1: init_geo per scene × n_views ─────────────────────────────────────
# Run once; skip if geometry already exists.
echo ""
echo "=========================================="
echo "Phase 1: Co-visible Geometry Initialisation"
echo "=========================================="

for N_VIEW in "${N_VIEWS[@]}"; do
    for DATASET_SCENE in "${SCENES[@]}"; do
        DATASET="${DATASET_SCENE%%:*}"
        SCENE="${DATASET_SCENE#*:}"
        SOURCE_PATH="${DATA_ROOT_DIR}/${DATASET}/${SCENE}/24_views/"
        MODEL_PATH="${OUTPUT_PPO_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views"

        mkdir -p "${MODEL_PATH}"

        # Skip if point cloud already initialised
        if [ -f "${MODEL_PATH}/sparse/0/points3D.bin" ] || [ -f "${MODEL_PATH}/sparse/0/points3D.txt" ]; then
            echo "Geometry already exists, skipping: ${MODEL_PATH}"
            continue
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] init_geo: ${DATASET}/${SCENE}  n_views=${N_VIEW}"
        CUDA_VISIBLE_DEVICES=0 python -W ignore ./init_geo.py \
            -s "${SOURCE_PATH}" \
            -m "${MODEL_PATH}" \
            --n_views "${N_VIEW}" \
            --focal_avg \
            --co_vis_dsp \
            --conf_aware_ranking \
            > "${MODEL_PATH}/01_init_geo.log" 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Log: ${MODEL_PATH}/01_init_geo.log"
    done
done

# ── Phase 2: one PPO run per n_views ──────────────────────────────────────────
echo ""
echo "=========================================="
echo "Phase 2: PPO Training"
echo "=========================================="

for N_VIEW in "${N_VIEWS[@]}"; do
    N_SCENES=${#SCENES[@]}
    NUM_EPISODES=$(( EPOCHS * N_SCENES ))
    PPO_MODEL_PATH="${OUTPUT_PPO_DIR}/ppo_nviews_${N_VIEW}"

    mkdir -p "${PPO_MODEL_PATH}"

    # Build the scene list: "source_path:model_path" pairs
    SCENE_PAIRS=()
    FIRST_SOURCE=""
    for DATASET_SCENE in "${SCENES[@]}"; do
        DATASET="${DATASET_SCENE%%:*}"
        SCENE="${DATASET_SCENE#*:}"
        SOURCE="${DATA_ROOT_DIR}/${DATASET}/${SCENE}/24_views/"
        MODEL="${OUTPUT_PPO_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views"
        SCENE_PAIRS+=( "${SOURCE}:${MODEL}" )
        if [ -z "${FIRST_SOURCE}" ]; then
            FIRST_SOURCE="${SOURCE}"
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PPO n_views=${N_VIEW}: ${N_SCENES} scenes × ${EPOCHS} epochs = ${NUM_EPISODES} episodes"

    CUDA_VISIBLE_DEVICES=0 python ./train_ppo.py \
        -s "${FIRST_SOURCE}" \
        -m "${PPO_MODEL_PATH}" \
        -r 1 \
        --n_views "${N_VIEW}" \
        --iterations "${gs_train_iter}" \
        --pp_optimizer \
        --optim_pose \
        --policy_backbone transformer \
        --train_state_encoder \
        --state_encoder_lr 1e-7 \
        --load_state_encoder "${STATE_ENCODER_CHECKPOINT}" \
        --load_ppo_policy "${PPO_POLICY_CHECKPOINT}" \
        --num_episodes "${NUM_EPISODES}" \
        --scene_list "${SCENE_PAIRS[@]}" \
        > "${PPO_MODEL_PATH}/02_train_ppo.log" 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PPO training done. Log: ${PPO_MODEL_PATH}/02_train_ppo.log"
done

# ── Phase 3: plot losses per n_views ──────────────────────────────────────────
echo ""
echo "=========================================="
echo "Phase 3: Plotting PPO Losses"
echo "=========================================="

for N_VIEW in "${N_VIEWS[@]}"; do
    PPO_MODEL_PATH="${OUTPUT_PPO_DIR}/ppo_nviews_${N_VIEW}"
    PLOTS_DIR="${OUTPUT_PPO_DIR}/plots/nviews_${N_VIEW}"

    mkdir -p "${PLOTS_DIR}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Plotting: n_views=${N_VIEW}"
    python plot_ppo_losses.py \
        --output_dir "${PPO_MODEL_PATH}" \
        --epochs "${EPOCHS}" \
        --plots_dir "${PLOTS_DIR}" \
        --scenes "${SCENES[@]}" \
        --n_views "${N_VIEW}"
done

echo ""
echo "=========================================="
echo "All PPO Training Complete!"
echo "Outputs : ${OUTPUT_PPO_DIR}"
echo "=========================================="
