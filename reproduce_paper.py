#!/usr/bin/env python3
"""
reproduce_paper.py
==================
End-to-end reproduction of the Table VI results on the Quiroga dataset:
preprocess -> train -> evaluate -> baselines, followed by a check of the
evaluation metrics against the values reported in the paper.

Requires the raw data at data/raw/quiroga_dataset.npy.
Tables VIII and IX are not covered (their scripts are not in this repository).
"""

import json
import subprocess
import sys
from pathlib import Path

# Values reported in the paper (Table VI)
EXPECTED = {
    "accuracy": {"ci_low": 0.962, "ci_high": 0.967},
    "precision": {"ci_low": 0.945, "ci_high": 0.952},
}
EXPECTED_LATENCY_MS = {"mean": 45.2, "tolerance": 3.0}  # amortized (batch of 32)

RAW_DATA = Path("data/raw/quiroga_dataset.npy")
DATA_DIR = Path("data/quiroga")
MODEL_DIR = Path("models/quiroga")
RESULTS_FILE = Path("results/reproduction/evaluation_results.json")


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print(f"Command: {cmd}")
    print("=" * 60)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    print("SUCCESS")
    return True


def main():
    if not RAW_DATA.exists():
        print(f"ERROR: raw data not found at {RAW_DATA}")
        sys.exit(1)

    steps = [
        ("Check dependencies",
         "python -c \"import tensorflow as tf; print(tf.__version__)\""),
        ("Preprocess Quiroga data",
         f"python preprocess.py --input {RAW_DATA} --output-dir {DATA_DIR} "
         f"--dataset-name quiroga"),
    ]
    for desc, cmd in steps:
        if not run_command(cmd, desc):
            print(f"\nREPRODUCTION FAILED at step: {desc}")
            sys.exit(1)

    data_file = DATA_DIR / "quiroga_preprocessed.npz"
    steps = [
        ("Train CNN-BiLSTM",
         f"python train.py --dataset quiroga --num-classes 6 --batch-size 32 "
         f"--model-dir {MODEL_DIR} --data-dir {DATA_DIR}"),
        ("Evaluate CNN-BiLSTM",
         f"python evaluate.py --model-weights {MODEL_DIR}/best_model_quiroga.h5 "
         f"--test-data {data_file} --num-classes 6 "
         f"--output-dir {RESULTS_FILE.parent}"),
        ("Run baselines",
         f"python baselines.py --data {data_file} --model all "
         f"--output-dir results/baselines"),
    ]
    for desc, cmd in steps:
        if not run_command(cmd, desc):
            print(f"\nREPRODUCTION FAILED at step: {desc}")
            sys.exit(1)

    # Validate results
    print(f"\n{'=' * 60}")
    print("VALIDATION: Checking results against paper (Table VI)")
    print("=" * 60)

    if not RESULTS_FILE.exists():
        print("ERROR: No evaluation results found")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    all_pass = True

    for metric, expected in EXPECTED.items():
        actual = results.get(metric, {}).get("point")
        if actual is None:
            print(f"MISSING: {metric}")
            all_pass = False
        elif expected["ci_low"] <= actual <= expected["ci_high"]:
            print(f"PASS: {metric} = {actual:.4f} "
                  f"(within [{expected['ci_low']}, {expected['ci_high']}])")
        else:
            print(f"FAIL: {metric} = {actual:.4f} "
                  f"(outside [{expected['ci_low']}, {expected['ci_high']}])")
            all_pass = False

    latency = results.get("latency", {}).get("batch_32", {}).get("mean_ms")
    if latency is None:
        print("MISSING: latency (batch_32)")
        all_pass = False
    elif abs(latency - EXPECTED_LATENCY_MS["mean"]) <= EXPECTED_LATENCY_MS["tolerance"]:
        print(f"PASS: latency = {latency:.1f} ms "
              f"(within ±{EXPECTED_LATENCY_MS['tolerance']} ms of {EXPECTED_LATENCY_MS['mean']})")
    else:
        print(f"FAIL: latency = {latency:.1f} ms "
              f"(outside ±{EXPECTED_LATENCY_MS['tolerance']} ms of {EXPECTED_LATENCY_MS['mean']})")
        all_pass = False

    print(f"\n{'=' * 60}")
    if all_pass:
        print("SUCCESS: All checked results match the paper within tolerance.")
        print("=" * 60)
        sys.exit(0)
    print("WARNING: Some results deviate from the paper.")
    print("Possible causes: different GPU, CUDA version, or random seed.")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
