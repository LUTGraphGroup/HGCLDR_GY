#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-paper-pilot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,0}"
OUTPUT_ROOT="${TUNING_OUTPUT_ROOT:-tuning-results}"
LOG_ROOT="${TUNING_LOG_ROOT:-tuning-logs}"
PARAM_DATASETS="${PARAM_DATASETS:-B-dataset,C-dataset,F-dataset}"
PARAM_FOLDS="${PARAM_FOLDS:-0,1,2}"
SENSITIVITY_FOLDS="${SENSITIVITY_FOLDS:-0}"
GRID_FOLDS="${GRID_FOLDS:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
SEED="${SEED:-1234}"
EVAL_FREQ="${EVAL_FREQ:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
PAPER_EPOCHS_B="${PAPER_EPOCHS_B:-1000}"
PAPER_EPOCHS_C="${PAPER_EPOCHS_C:-2000}"
PAPER_EPOCHS_F="${PAPER_EPOCHS_F:-1000}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ "${#GPUS[@]}" -gt 0 ]] || {
  echo "No GPU IDs supplied." >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

declare -a WORKER_PIDS=()
cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

dataset_params() {
  local dataset="$1"
  case "$dataset" in
    B-dataset)
      PAPER_EPOCHS="$PAPER_EPOCHS_B"
      PAPER_SSL_REG="0.01"
      ;;
    C-dataset)
      PAPER_EPOCHS="$PAPER_EPOCHS_C"
      PAPER_SSL_REG="0.10"
      ;;
    F-dataset)
      PAPER_EPOCHS="$PAPER_EPOCHS_F"
      PAPER_SSL_REG="0.10"
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      return 2
      ;;
  esac
}

parse_datasets() {
  IFS=',' read -r -a PARSED_DATASETS <<< "$PARAM_DATASETS"
  local dataset
  [[ "${#PARSED_DATASETS[@]}" -gt 0 ]] || {
    echo "No datasets supplied through PARAM_DATASETS." >&2
    return 2
  }
  for dataset in "${PARSED_DATASETS[@]}"; do
    dataset_params "$dataset" >/dev/null
  done
}

parse_folds() {
  local value="$1"
  IFS=',' read -r -a PARSED_FOLDS <<< "$value"
  local fold
  for fold in "${PARSED_FOLDS[@]}"; do
    [[ "$fold" =~ ^[0-9]+$ ]] && ((fold >= 0 && fold <= 9)) || {
      echo "Invalid fold '$fold'; expected comma-separated values from 0 to 9." >&2
      return 2
    }
  done
}

value_id() {
  local value="$1"
  printf '%s' "${value//./p}"
}

run_one() {
  local gpu="$1" dataset="$2" fold="$3" epochs="$4"
  local diffusion_steps="$5" layers="$6" ssl_ratio="$7"
  local ssl_reg="$8" ssl_temp="$9" tag="${10}"
  local fold_name job_id result_file stdout_log
  fold_name="fold_$(printf '%02d' "$fold")"
  job_id="${tag}_${dataset}_${fold_name}"
  result_file="${OUTPUT_ROOT}/${tag}/${dataset}/${fold_name}/selection_metrics.json"
  stdout_log="${LOG_ROOT}/${job_id}.log"

  if [[ -s "$result_file" ]]; then
    echo "[SKIP] $job_id already completed"
    return 0
  fi
  if [[ "$PLAN_ONLY" == "1" ]]; then
    echo "[PLAN] gpu=$gpu job=$job_id epochs=$epochs T=$diffusion_steps L=$layers r_ssl=$ssl_ratio lambda_ssl=$ssl_reg tau=$ssl_temp"
    return 0
  fi

  echo "[START] $(date '+%F %T') gpu=$gpu job=$job_id"
  if ! PROJ_LGCN_K="$diffusion_steps" CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" -u run.py \
      --dataset "$dataset" \
      --fold "$fold" \
      --pseudo_mode none \
      --selection_only 1 \
      --use_fixed_validation_pairs 1 \
      --refit_after_selection 0 \
      --epochs "$epochs" \
      --eval-freq "$EVAL_FREQ" \
      --batch-size "$BATCH_SIZE" \
      --embedding_dim 32 \
      --num-layers "$layers" \
      --lr 0.001 \
      --weight-decay 0.005 \
      --momentum 0.95 \
      --max_norm 1.5 \
      --ssl_ratio "$ssl_ratio" \
      --ssl_reg "$ssl_reg" \
      --ssl_temp "$ssl_temp" \
      --num_neg 8 \
      --seed "$SEED" \
      --device cuda:0 \
      --output_root "$OUTPUT_ROOT" \
      --run_tag "$tag" \
      --log "$job_id" >"$stdout_log" 2>&1; then
    echo "[FAIL] $(date '+%F %T') gpu=$gpu job=$job_id exited non-zero; inspect $stdout_log" >&2
    return 1
  fi

  [[ -s "$result_file" ]] || {
    echo "[FAIL] $job_id produced no selection metrics; inspect $stdout_log" >&2
    return 1
  }
  echo "[DONE] $(date '+%F %T') gpu=$gpu job=$job_id"
}

run_queue() {
  local gpu="$1" queue="$2"
  local dataset fold epochs diffusion_steps layers ssl_ratio ssl_reg ssl_temp tag
  local failed=0
  while IFS=',' read -r dataset fold epochs diffusion_steps layers ssl_ratio ssl_reg ssl_temp tag; do
    [[ -z "${dataset:-}" ]] && continue
    if ! run_one "$gpu" "$dataset" "$fold" "$epochs" "$diffusion_steps" \
      "$layers" "$ssl_ratio" "$ssl_reg" "$ssl_temp" "$tag"; then
      failed=1
    fi
  done <<< "$queue"
  return "$failed"
}

run_jobs() {
  local -a jobs=("$@") queues=()
  local n_gpu="${#GPUS[@]}" i slot pid failed=0
  for ((i=0; i<n_gpu; i++)); do
    queues[$i]=""
  done
  for ((i=0; i<${#jobs[@]}; i++)); do
    slot=$((i % n_gpu))
    queues[$slot]+="${jobs[$i]}"$'\n'
  done

  WORKER_PIDS=()
  for ((i=0; i<n_gpu; i++)); do
    [[ -z "${queues[$i]}" ]] && continue
    run_queue "${GPUS[$i]}" "${queues[$i]}" &
    WORKER_PIDS+=("$!")
  done
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || failed=1
  done
  WORKER_PIDS=()
  [[ "$failed" -eq 0 ]] || {
    echo "At least one tuning job failed." >&2
    return 1
  }
}

build_smoke_jobs() {
  local dataset
  parse_datasets
  JOBS=()
  for dataset in "${PARSED_DATASETS[@]}"; do
    dataset_params "$dataset"
    JOBS+=("$dataset,0,$SMOKE_EPOCHS,3,4,0.05,$PAPER_SSL_REG,0.05,tune_parameter_smoke")
  done
}

build_paper_pilot_jobs() {
  local dataset fold
  parse_datasets
  parse_folds "$PARAM_FOLDS"
  JOBS=()
  for dataset in "${PARSED_DATASETS[@]}"; do
    dataset_params "$dataset"
    for fold in "${PARSED_FOLDS[@]}"; do
      JOBS+=("$dataset,$fold,$PAPER_EPOCHS,3,4,0.05,$PAPER_SSL_REG,0.05,tune_paper_baseline")
    done
  done
}

build_sensitivity_jobs() {
  local dataset fold value id
  parse_datasets
  parse_folds "$SENSITIVITY_FOLDS"
  JOBS=()
  for dataset in "${PARSED_DATASETS[@]}"; do
    dataset_params "$dataset"
    for fold in "${PARSED_FOLDS[@]}"; do
      for value in 1 2 3 4 5; do
        JOBS+=("$dataset,$fold,$PAPER_EPOCHS,$value,4,0.05,$PAPER_SSL_REG,0.05,tune_T_$value")
      done
      for value in 2 3 4 5 6; do
        JOBS+=("$dataset,$fold,$PAPER_EPOCHS,3,$value,0.05,$PAPER_SSL_REG,0.05,tune_L_$value")
      done
      for value in 0.05 0.10 0.15 0.20; do
        id="$(value_id "$value")"
        JOBS+=("$dataset,$fold,$PAPER_EPOCHS,3,4,$value,$PAPER_SSL_REG,0.05,tune_rssl_$id")
      done
    done
  done
}

build_cross_grid_jobs() {
  local dataset fold ssl_reg ssl_temp reg_id temp_id
  parse_datasets
  parse_folds "$GRID_FOLDS"
  JOBS=()
  for dataset in "${PARSED_DATASETS[@]}"; do
    dataset_params "$dataset"
    for fold in "${PARSED_FOLDS[@]}"; do
      for ssl_reg in 0.001 0.005 0.01 0.05 0.10; do
        reg_id="$(value_id "$ssl_reg")"
        for ssl_temp in 0.01 0.05 0.10 0.20 0.50; do
          temp_id="$(value_id "$ssl_temp")"
          JOBS+=("$dataset,$fold,$PAPER_EPOCHS,3,4,0.05,$ssl_reg,$ssl_temp,tune_grid_reg_${reg_id}_temp_${temp_id}")
        done
      done
    done
  done
}

summarize() {
  [[ "$PLAN_ONLY" == "1" ]] && return 0
  "$PYTHON_BIN" scripts/summarize_parameter_selection.py \
    --root "$OUTPUT_ROOT"
}

case "$STAGE" in
  smoke)
    build_smoke_jobs
    run_jobs "${JOBS[@]}"
    ;;
  paper-pilot)
    build_paper_pilot_jobs
    run_jobs "${JOBS[@]}"
    ;;
  sensitivity)
    build_sensitivity_jobs
    run_jobs "${JOBS[@]}"
    ;;
  cross-grid)
    build_cross_grid_jobs
    run_jobs "${JOBS[@]}"
    ;;
  paper-figures)
    build_sensitivity_jobs
    run_jobs "${JOBS[@]}"
    build_cross_grid_jobs
    run_jobs "${JOBS[@]}"
    ;;
  all)
    build_paper_pilot_jobs
    run_jobs "${JOBS[@]}"
    build_sensitivity_jobs
    run_jobs "${JOBS[@]}"
    build_cross_grid_jobs
    run_jobs "${JOBS[@]}"
    ;;
  *)
    echo "Use smoke|paper-pilot|sensitivity|cross-grid|paper-figures|all" >&2
    exit 2
    ;;
esac

summarize
echo "Stage '$STAGE' completed successfully."
