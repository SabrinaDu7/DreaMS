#!/bin/bash
#SBATCH --job-name=DreaMS_pre-training
#SBATCH --account=def-hsn
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sabrina7/scratch/dreams_runs/DreaMS_pre-training_result_%j.out
#SBATCH --error=/home/sabrina7/scratch/dreams_runs/DreaMS_pre-training_result_%j.err
#SBATCH --mail-user=sabrina.du@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Avoid a cluster-loaded Python/SciPy module's PYTHONPATH leaking a different
# numpy into the venv's Python and breaking rdkit's compiled ABI
# (`_ARRAY_API not found`).
unset PYTHONPATH

# --- Config ---------------------------------------------------------------
# Persistent clone of this repo.
REPO_DIR="${HOME}/experiments/DreaMS"

# Persistent location of the pre-training dataset (lives outside the repo).
DATASET_SRC="${SCRATCH}/datasets/gems/GeMS_A10.hdf5"

# Fast node-local copies used for the run.
RUN_REPO_DIR="${SLURM_TMPDIR}/DreaMS"
RUN_DATASET_PTH="${SLURM_TMPDIR}/GeMS_A10.hdf5"

project_name="dreams"
job_key="${SLURM_JOB_ID}_$(date +'%m-%d_%H-%M')"
max_epochs="${1:-3000}"

WANDB_API_KEY_FILE="${REPO_DIR}/wandb_api.txt"
if [ ! -s "${WANDB_API_KEY_FILE}" ]; then
    echo "Wandb API key file not found or empty: ${WANDB_API_KEY_FILE}" >&2
    exit 1
fi

# --- Copy repo and dataset to node-local storage for speed ---
rsync -a --exclude='.git' --exclude='.venv' "${REPO_DIR}/" "${RUN_REPO_DIR}/"
cp "${DATASET_SRC}" "${RUN_DATASET_PTH}"
cd "${RUN_REPO_DIR}"

# --- Build env locally and activate it ---
# Keep uv's cache on the same node-local filesystem as the venv so it can
# hardlink instead of copying package files (faster, avoids the warning below).
export UV_CACHE_DIR="${SLURM_TMPDIR}/uv-cache"
uv sync --frozen
source .venv/bin/activate

# --- Export project definitions ---
$(python -c "from dreams.definitions import export; export()")

# --- wandb auth (non-interactive, key read from file) ---
wandb login "$(cat "${WANDB_API_KEY_FILE}")"

# --- Sync checkpoints back to scratch, periodically and on exit ---
# ModelCheckpoint (train.py) writes every 1000 steps with save_top_k=-1 (keeps all),
# under a path relative to cwd, so it lands in ${RUN_REPO_DIR}/dreams/${project_name}/${job_key}.
CKPT_DIR="${DREAMS_DIR}/${job_key}"
CKPT_DIR_SCRATCH="${SCRATCH}/dreams_runs/${job_key}"
mkdir -p "${CKPT_DIR_SCRATCH}"

sync_back() {
    rsync -a "${CKPT_DIR}/" "${CKPT_DIR_SCRATCH}/" 2>/dev/null || true
}

(
    while true; do
        sleep 1200  # 20 min
        sync_back
    done
) &
sync_loop_pid=$!

cleanup() {
    kill "${sync_loop_pid}" 2>/dev/null || true
    echo "Syncing checkpoints back to ${CKPT_DIR_SCRATCH}..."
    sync_back
}
trap cleanup EXIT TERM INT

# Move to running dir
cd "${DREAMS_DIR}" || exit 3

# --- Run the training script ---
# For an interactive smoke test before submitting: salloc a GPU node (SLURM_TMPDIR
# is set there too), run everything above by hand, then run this same command with
# --max_epochs 1 and --num_devices set to whatever your interactive allocation has.
#
# Replace `python3 training/train.py` with `srun --export=ALL --preserve-env python3 training/train.py \`
# when executing on a SLURM cluster via `sbatch`.
uv run training/train.py \
 --project_name "${project_name}" \
 --job_key "${job_key}" \
 --run_name "${job_key}" \
 --frac_masks 0.3 \
 --train_regime pre-training \
 --dataset_pth "${RUN_DATASET_PTH}" \
 --val_check_interval 0.1 \
 --train_objective mask_mz_hot \
 --hot_mz_bin_size 0.05 \
 --dformat A \
 --model DreaMS \
 --ff_peak_depth 1 \
 --ff_fourier_depth 5 \
 --ff_fourier_d 512 \
 --ff_out_depth 1 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --num_workers_data 8 \
 --max_epochs "${max_epochs}" \
 --log_every_n_steps 20 \
 --seed 3402 \
 --n_layers 7 \
 --n_heads 8 \
 --d_peak 44 \
 --d_fourier 980 \
 --lr 1e-4 \
 --batch_size 256 \
 --dropout 0.1 \
 --save_top_k -1 \
 --att_dropout 0.1 \
 --residual_dropout 0.1 \
 --ff_dropout 0.1 \
 --weight_decay 0 \
 --attn_mech dot-product \
 --train_precision 32 \
 --mask_peaks \
 --mask_intens_strategy intens_p \
 --max_peaks_n 60 \
 --ssl_probing_depth 0 \
 --focal_loss_gamma 5 \
 --no_transformer_bias \
 --n_warmup_steps 5000 \
 --fourier_strategy lin_float_int \
 --mz_shift_aug_p 0.2 \
 --mz_shift_aug_max 50 \
 --pre_norm \
 --graphormer_mz_diffs \
 --ret_order_loss_w 0.2 \
 --wandb_entity_name sabrina-du-mcgill-university \
