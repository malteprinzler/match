#!/bin/bash
# Usage: ./seg_ava256.sh [INPUT] [OUTPUT] [FRAME_STRIDE] [SUBJECTS] [CAMERAS] 2>&1 | tee seg_ava256.log
# SUBJECTS and CAMERAS should be space-separated strings if provided.

# Default values
DEFAULT_INPUT='data/ava-256'
DEFAULT_OUTPUT="data/sapiens_segmentations/ava-256/framestride_10"
DEFAULT_FRAME_STRIDE=10
DEFAULT_SUBJECTS=""  # e.g. "INQ807 KWL586 PDG961 LCJ763 UHV563 PGO261 IBQ026 APP152 YJF815 PSV686 TCE049"
DEFAULT_CAMERAS=""  # e.g. "400939 400943 400948 400951 400953 400981 400983 401033 401036 401037 401042 401046 401067 401071 401072 401077 401078 401131 401158 401162 401163 401166 401168 401175 401292 401294 401301 401312 401313 401316 401383 401399 401404 401413 401454 401457 401458 401459 401466"

# Assign arguments or defaults
INPUT=${1:-$DEFAULT_INPUT}
OUTPUT=${2:-$DEFAULT_OUTPUT}
FRAME_STRIDE=${3:-$DEFAULT_FRAME_STRIDE}
SUBJECTS_STR=${4:-$DEFAULT_SUBJECTS}
CAMERAS=${5:-$DEFAULT_CAMERAS}

# Convert SUBJECTS_STR to array
SUBJECTS=($SUBJECTS_STR)

SAPIENS_CHECKPOINT_ROOT=assets/sapiens/checkpoints

# MODE='torchscript' ## original. no optimizations (slow). full precision inference.
MODE='bfloat16' ## A100 gpus. faster inference at bfloat16

SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_$MODE.pt2
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_$MODE.pt2
MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_$MODE.pt2

OUTPUT=$OUTPUT/$MODEL_NAME

##-------------------------------------inference-------------------------------------
RUN_FILE='third_party/sapiens/lite/demo/vis_seg_ava.py'
BATCH_SIZE=10

# Find all images and sort them, then write to a temporary text file
IMAGE_LIST="${INPUT}/image_list_${CONDOR_ClusterId}_${CONDOR_Process}.txt"

if [ ${#SUBJECTS[@]} -eq 0 ]; then
  find "${INPUT}" -wholename '*/decoder/image/cam*.zip' | sort > "${IMAGE_LIST}"  # AVA DEFAULT
else
  # Specific subjects only
  FIND_CMD="find \"${INPUT}\""
  for X in "${SUBJECTS[@]}"; do
    FIND_CMD+=" -path '*--${X}/decoder/image/cam*.zip' -o"
  done
  # remove trailing -o
  FIND_CMD="${FIND_CMD% -o}"
  FIND_CMD+=" | sort > \"${IMAGE_LIST}\""
  eval "$FIND_CMD"
fi


# # Instead use an existing image list file (e.g. from a failed previous job)
# IMAGE_LIST_='/fast/mprinzler/gintern/datasets/ava-256/myimage_list_16374907_10.txt'
# cp $IMAGE_LIST_ $IMAGE_LIST

# Check if image list was created successfully
echo "${IMAGE_LIST}"
if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found. Check your input directory and permissions."
  exit 1
fi

# Count images and calculate the number of images per text file
NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
TOTAL_JOBS="${CONDOR_WORLD_SIZE:-1}"
JOB_IDX="${CONDOR_Process:-0}"
IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))

export TF_CPP_MIN_LOG_LEVEL=2
# echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

# Divide image paths into text files for each job
i="${JOB_IDX}"
TEXT_FILE="${INPUT}/myimage_list_${CONDOR_ClusterId}_${JOB_IDX}.txt"
if [ $i -eq $((TOTAL_JOBS - 1)) ]; then
  # For the last text file, write all remaining image paths
  tail -n +$((IMAGES_PER_FILE * i + 1)) "${IMAGE_LIST}" > "${TEXT_FILE}"
else
  # Write the exact number of image paths per text file
  head -n $((IMAGES_PER_FILE * (i + 1))) "${IMAGE_LIST}" | tail -n ${IMAGES_PER_FILE} > "${TEXT_FILE}"
fi


if [ -n "$CAMERAS" ]; then
  python ${RUN_FILE} \
    ${CHECKPOINT} \
    --input "${TEXT_FILE}" \
    --batch-size="${BATCH_SIZE}" \
    --cameras ${CAMERAS} \
    --frame_stride="${FRAME_STRIDE}" \
    --output-root="${OUTPUT}"
else
  # RUN ON ALL CAMERAS
  python ${RUN_FILE} \
    ${CHECKPOINT} \
    --input "${TEXT_FILE}" \
    --batch-size="${BATCH_SIZE}" \
    --frame_stride="${FRAME_STRIDE}" \
    --output-root="${OUTPUT}"
fi

# Wait for all background processes to finish
wait

# Remove the image list and temporary text files
rm "${IMAGE_LIST}"
rm "${TEXT_FILE}"

# Go back to the original script's directory
cd -

echo "Processing complete."
echo "Results saved to $OUTPUT"
