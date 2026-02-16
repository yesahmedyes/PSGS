#!/bin/bash

# Change the absolute path first!
DATA_ROOT_DIR="/home/ubuntu/ahmed-etri"

OUTPUT_DIR="output_se_training"
EPOCHS=20

DATASETS=(
    tanks_templates
    # MVimgNet
)

SCENES=(
    Horse
    Family
    Church
    Barn
    Ballroom
    Francis
    Ignatius
    Museum
)

N_VIEWS=(
    3
    6
    12
)

gs_train_iter=(
    1000
    2000
)

# State encoder checkpoint location
GLOBAL_CHECKPOINT="pggs/checkpoints/state_encoder_final.pth"

# Function: Train scene with state encoder
train_scene_with_state_encoder() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    local EPOCH=$5
    local LOAD_CHECKPOINT=$6  # Path to checkpoint to load (empty for epoch 1, scene 1)
    
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/24_views/
    GT_POSE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    IMAGE_PATH=${SOURCE_PATH}images
    MODEL_PATH=${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views/epoch_${EPOCH}

    # Create necessary directories
    mkdir -p ${MODEL_PATH}

    echo "======================================================="
    echo "Training: ${DATASET}/${SCENE} (${N_VIEW} views) - Epoch ${EPOCH}"
    echo "GPU: ${GPU_ID}"
    if [ -n "$LOAD_CHECKPOINT" ]; then
        echo "Loading checkpoint: ${LOAD_CHECKPOINT}"
    else
        echo "Starting from random initialization"
    fi
    echo "======================================================="

    # (1) Co-visible Global Geometry Initialization
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW} \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    > ${MODEL_PATH}/01_init_geo.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"
 
    # (2) Train with state encoder
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training with state encoder..."
    
    # Build the command
    TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train_se.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEW} \
    --iterations ${gs_train_iter} \
    --pp_optimizer \
    --optim_pose"
    
    # Add checkpoint loading if available
    if [ -n "$LOAD_CHECKPOINT" ] && [ -f "$LOAD_CHECKPOINT" ]; then
        TRAIN_CMD="${TRAIN_CMD} --load_state_encoder ${LOAD_CHECKPOINT}"
    fi
    
    # Execute training
    eval "${TRAIN_CMD} > ${MODEL_PATH}/02_train_se.log 2>&1"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train_se.log"

    echo "======================================================="
    echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views) - Epoch ${EPOCH}"
    echo "======================================================="
}

# Main training loop
echo "=========================================="
echo "State Encoder Multi-Epoch Training"
echo "Total Epochs: ${EPOCHS}"
echo "Scenes: ${SCENES[@]}"
echo "=========================================="

# Track current global checkpoint
CURRENT_CHECKPOINT=""
FIRST_SCENE=true

for EPOCH in $(seq 1 $EPOCHS); do
    echo ""
    echo "=========================================="
    echo "EPOCH ${EPOCH}/${EPOCHS}"
    echo "=========================================="
    
    for DATASET in "${DATASETS[@]}"; do
        for SCENE in "${SCENES[@]}"; do
            for N_VIEW in "${N_VIEWS[@]}"; do
                for gs_train_iter in "${gs_train_iter[@]}"; do
                    # Determine which checkpoint to load
                    if [ "$FIRST_SCENE" = true ]; then
                        # First scene of first epoch - no checkpoint
                        LOAD_CHKPT=""
                        FIRST_SCENE=false
                    else
                        # All other scenes - load from previous scene
                        LOAD_CHKPT="${CURRENT_CHECKPOINT}"
                    fi
                    
                    # Train scene (sequential execution - GPU 0)
                    train_scene_with_state_encoder \
                        0 \
                        "$DATASET" \
                        "$SCENE" \
                        "$N_VIEW" \
                        "$EPOCH" \
                        "$LOAD_CHKPT"
                    
                    # Update global checkpoint to the one just saved
                    CURRENT_CHECKPOINT="${GLOBAL_CHECKPOINT}"
                done
            done
        done
    done
done

# Generate plots
echo ""
echo "=========================================="
echo "Generating Loss Plots"
echo "=========================================="

PLOTS_DIR="${OUTPUT_DIR}/plots"
mkdir -p ${PLOTS_DIR}

python scripts/plot_state_encoder_losses.py \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --plots_dir ${PLOTS_DIR} \
    --scenes ${SCENES[@]}

echo ""
echo "=========================================="
echo "All Training Complete!"
echo "Plots saved to: ${PLOTS_DIR}"
echo "=========================================="
