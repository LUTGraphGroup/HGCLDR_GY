#!/usr/bin/env bash
set -Eeuo pipefail

# Stages include the legacy non-refit experiments, the corrected refit main
# experiment, and the corrected hard/weighted pseudo-supervision ablation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
MAIN_EPOCHS="${MAIN_EPOCHS:-1000}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
REFIT_EPOCHS_B="${REFIT_EPOCHS_B:-2000}"
REFIT_EPOCHS_C="${REFIT_EPOCHS_C:-2000}"
REFIT_EPOCHS_F="${REFIT_EPOCHS_F:-2000}"
SSL_REG_B="${SSL_REG_B:-0.01}"
SSL_REG_C="${SSL_REG_C:-0.10}"
SSL_REG_F="${SSL_REG_F:-0.10}"
SEED="${SEED:-1234}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
LOG_ROOT="${LOG_ROOT:-server-logs}"
EVAL_FREQ="${EVAL_FREQ:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
REFIT_PSEUDO_MODES="${REFIT_PSEUDO_MODES:-hard,weighted}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "No GPU IDs supplied." >&2; exit 2; }
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

declare -a WORKER_PIDS=()
cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM

run_one() {
  local gpu="$1" mode="$2" dataset="$3" fold="$4" epochs="$5" ssl_reg="$6" refit="$7" tag="$8"
  local fold_name job_id result_file stdout_log
  fold_name="fold_$(printf '%02d' "$fold")"
  job_id="${tag}_${dataset}_${fold_name}"
  result_file="${OUTPUT_ROOT}/${tag}/${dataset}/${fold_name}/metrics.json"
  stdout_log="${LOG_ROOT}/${job_id}.log"

  if [[ -s "$result_file" ]]; then
    echo "[SKIP] $job_id already completed"
    return 0
  fi

  echo "[START] $(date '+%F %T') gpu=$gpu job=$job_id"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u run.py \
    --dataset "$dataset" --fold "$fold" --pseudo_mode "$mode" \
    --epochs "$epochs" --eval-freq "$EVAL_FREQ" --batch-size "$BATCH_SIZE" \
    --ssl_reg "$ssl_reg" --refit_after_selection "$refit" \
    --seed "$SEED" --device cuda:0 --output_root "$OUTPUT_ROOT" \
    --run_tag "$tag" --log "$job_id" >"$stdout_log" 2>&1

  [[ -s "$result_file" ]] || {
    echo "[FAIL] $job_id produced no metrics; inspect $stdout_log" >&2
    return 1
  }
  echo "[DONE] $(date '+%F %T') gpu=$gpu job=$job_id"
}

run_queue() {
  local gpu="$1" queue="$2" mode dataset fold epochs ssl_reg refit tag
  while IFS=',' read -r mode dataset fold epochs ssl_reg refit tag; do
    [[ -z "${mode:-}" ]] && continue
    run_one "$gpu" "$mode" "$dataset" "$fold" "$epochs" "$ssl_reg" "$refit" "$tag"
  done <<< "$queue"
}

run_jobs() {
  local -a jobs=("$@") queues=()
  local n_gpu="${#GPUS[@]}" i slot pid failed=0
  for ((i=0; i<n_gpu; i++)); do queues[$i]=""; done
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
  for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || failed=1; done
  WORKER_PIDS=()
  [[ "$failed" -eq 0 ]] || {
    echo "At least one job failed; later stages were not started." >&2
    return 1
  }
}

build_smoke_jobs() {
  local dataset
  JOBS=()
  for dataset in B-dataset C-dataset F-dataset; do
    JOBS+=("none,$dataset,0,$SMOKE_EPOCHS,0.005,0,smoke_none")
  done
}

build_pilot_jobs() {
  local dataset fold
  JOBS=()
  for dataset in B-dataset C-dataset F-dataset; do
    for fold in 0 1 2; do
      JOBS+=("none,$dataset,$fold,$MAIN_EPOCHS,0.005,0,main_none")
    done
  done
}

build_main_jobs() {
  local dataset fold
  JOBS=()
  for dataset in B-dataset C-dataset F-dataset; do
    for fold in {0..9}; do
      JOBS+=("none,$dataset,$fold,$MAIN_EPOCHS,0.005,0,main_none")
    done
  done
}

build_ablation_jobs() {
  local mode dataset fold
  JOBS=()
  for mode in hard weighted; do
    for dataset in B-dataset C-dataset F-dataset; do
      for fold in {0..9}; do
        JOBS+=("$mode,$dataset,$fold,$MAIN_EPOCHS,0.005,0,pseudo_$mode")
      done
    done
  done
}

refit_params() {
  local dataset="$1"
  case "$dataset" in
    B-dataset) REFIT_MAX_EPOCHS="$REFIT_EPOCHS_B"; REFIT_SSL_REG="$SSL_REG_B" ;;
    C-dataset) REFIT_MAX_EPOCHS="$REFIT_EPOCHS_C"; REFIT_SSL_REG="$SSL_REG_C" ;;
    F-dataset) REFIT_MAX_EPOCHS="$REFIT_EPOCHS_F"; REFIT_SSL_REG="$SSL_REG_F" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 2 ;;
  esac
}

build_refit_smoke_jobs() {
  local dataset
  JOBS=()
  for dataset in B-dataset C-dataset F-dataset; do
    refit_params "$dataset"
    JOBS+=("none,$dataset,0,$SMOKE_EPOCHS,$REFIT_SSL_REG,1,refit_smoke_none")
  done
}

build_refit_pilot_jobs() {
  local dataset fold
  JOBS=()
  for dataset in B-dataset C-dataset F-dataset; do
    refit_params "$dataset"
    for fold in 0 1 2; do
      JOBS+=("none,$dataset,$fold,$REFIT_MAX_EPOCHS,$REFIT_SSL_REG,1,refit_none_v1")
    done
  done
}

build_refit_main_jobs() {
  local dataset fold
  JOBS=()
  for dataset in B-dataset C-dataset F-dataset; do
    refit_params "$dataset"
    for fold in {0..9}; do
      JOBS+=("none,$dataset,$fold,$REFIT_MAX_EPOCHS,$REFIT_SSL_REG,1,refit_none_v1")
    done
  done
}

build_refit_pseudo_jobs() {
  local mode dataset fold
  local -a pseudo_modes
  IFS=',' read -r -a pseudo_modes <<< "$REFIT_PSEUDO_MODES"
  JOBS=()
  for mode in "${pseudo_modes[@]}"; do
    case "$mode" in
      hard|weighted) ;;
      *) echo "Unsupported refit pseudo mode: $mode (use hard or weighted)" >&2; return 2 ;;
    esac
    for dataset in B-dataset C-dataset F-dataset; do
      refit_params "$dataset"
      for fold in {0..9}; do
        JOBS+=("$mode,$dataset,$fold,$REFIT_MAX_EPOCHS,$REFIT_SSL_REG,1,refit_pseudo_${mode}_v1")
      done
    done
  done
}

assert_refit_pseudo_complete() {
  local mode dataset fold result_file missing=0
  local -a pseudo_modes
  IFS=',' read -r -a pseudo_modes <<< "$REFIT_PSEUDO_MODES"
  for mode in "${pseudo_modes[@]}"; do
    case "$mode" in
      hard|weighted) ;;
      *) echo "Unsupported refit pseudo mode: $mode (use hard or weighted)" >&2; return 2 ;;
    esac
    for dataset in B-dataset C-dataset F-dataset; do
      for fold in {0..9}; do
        result_file="${OUTPUT_ROOT}/refit_pseudo_${mode}_v1/${dataset}/fold_$(printf '%02d' "$fold")/metrics.json"
        if [[ ! -s "$result_file" ]]; then
          echo "[MISSING] $result_file" >&2
          missing=1
        fi
      done
    done
  done
  [[ "$missing" -eq 0 ]] || {
    echo "Fixed-pair rescoring requires every requested training result." >&2
    return 1
  }
}

rescore_refit_pseudo() {
  local mode
  local -a pseudo_modes
  IFS=',' read -r -a pseudo_modes <<< "$REFIT_PSEUDO_MODES"
  assert_refit_pseudo_complete
  for mode in "${pseudo_modes[@]}"; do
    echo "[RESCORE] $(date '+%F %T') mode=$mode fixed validation/test pairs"
    "$PYTHON_BIN" scripts/rescore_fixed_eval.py \
      --source-root "${OUTPUT_ROOT}/refit_pseudo_${mode}_v1" \
      --output-root "$OUTPUT_ROOT" \
      --run-tag "refit_pseudo_${mode}_fixed_eval_v1" \
      --device cuda:0 \
      --folds 0 1 2 3 4 5 6 7 8 9
  done
}

summarize() {
  "$PYTHON_BIN" scripts/summarize_results.py --root "$OUTPUT_ROOT" || true
}

case "$STAGE" in
  smoke) build_smoke_jobs; run_jobs "${JOBS[@]}" ;;
  pilot) build_pilot_jobs; run_jobs "${JOBS[@]}" ;;
  main) build_main_jobs; run_jobs "${JOBS[@]}" ;;
  ablation) build_ablation_jobs; run_jobs "${JOBS[@]}" ;;
  refit-smoke) build_refit_smoke_jobs; run_jobs "${JOBS[@]}" ;;
  refit-pilot) build_refit_pilot_jobs; run_jobs "${JOBS[@]}" ;;
  refit-main) build_refit_main_jobs; run_jobs "${JOBS[@]}" ;;
  refit-pseudo) build_refit_pseudo_jobs; run_jobs "${JOBS[@]}" ;;
  rescore-refit-pseudo) rescore_refit_pseudo ;;
  refit-pseudo-all)
    build_refit_pseudo_jobs; run_jobs "${JOBS[@]}"
    rescore_refit_pseudo
    ;;
  all-main)
    build_smoke_jobs; run_jobs "${JOBS[@]}"
    build_pilot_jobs; run_jobs "${JOBS[@]}"
    build_main_jobs; run_jobs "${JOBS[@]}"
    ;;
  *) echo "Use smoke|pilot|main|ablation|all-main|refit-smoke|refit-pilot|refit-main|refit-pseudo|rescore-refit-pseudo|refit-pseudo-all" >&2; exit 2 ;;
esac

summarize
echo "Stage '$STAGE' completed successfully."
