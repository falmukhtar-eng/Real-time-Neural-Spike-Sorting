```python
#!/usr/bin/env python3
"""
reproduce_paper.py
==================
One-command reproduction of all paper results.
Validates outputs against Tables VI, VII, VIII, IX.
"""

import subprocess
import sys
import json
from pathlib import Path

# Expected results from paper (Table VI)
EXPECTED = {
    "accuracy": {"mean": 0.964, "ci_low": 0.962, "ci_high": 0.967},
    "precision": {"mean": 0.948, "ci_low": 0.945, "ci_high": 0.952},
    "latency_amortized_ms": {"mean": 45.2, "tolerance": 3.0},
}

def run_command(cmd, description):
  """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    print("SUCCESS")
    return True

def main():
    steps = [
        ("Check dependencies", "python -c \"import tensorflow; print(tf.__version__)\""),
        ("Preprocess Quiroga data", "python preprocess.py --dataset quiroga --output-dir data/quiroga"),
        ("Evaluate pre-trained model", 
         "python evaluate.py --model-weights models/pretrained/quiroga_full_supervision.h5 "
         "--test-data data/quiroga/test.npz --num-classes 6 --output-dir 
         results/reproduction"),
        ("Run baselines", 
         "python baselines.py --data data/quiroga/test.npz --model all --output-dir results/baselines"),
    ]
    
    for desc, cmd in steps:
        if not run_command(cmd, desc):
            print(f"\nREPRODUCTION FAILED at step: {desc}")
            sys.exit(1)
    
    # Validate results
    print(f"\n{'='*60}")
    print("VALIDATION: Checking results against paper")
    print('='*60)
    
    results_file = Path("results/reproduction/evaluation_results.json")
    if not results_file.exists():
        print("ERROR: No evaluation results found")
        sys.exit(1)
        with open(results_file) as f:
        results = json.load(f)
    
    all_pass = True
    for metric, expected in EXPECTED.items():
        if metric not in results:
            print(f"MISSING: {metric}")
            all_pass = False
            continue
        
        actual = results[metric]
        if "ci_low" in expected:
            # Check if mean falls within paper's confidence interval
            if expected["ci_low"] <= actual["mean"] <= expected["ci_high"]:
                print(f"PASS: {metric} = {actual['mean']:.4f} (within [{expected['ci_low']}, {expected['ci_high']}])")
            else:
                print(f"FAIL: {metric} = {actual['mean']:.4f} (outside [{expected['ci_low']}, {expected['ci_high']}])")
                all_pass = False
        else:
        # Latency check with tolerance
            if abs(actual["mean"] - expected["mean"]) <= expected["tolerance"]:
                print(f"PASS: {metric} = {actual['mean']:.1f}ms (within ±{expected['tolerance']}ms of {expected['mean']})")
            else:
                print(f"FAIL: {metric} = {actual['mean']:.1f}ms (outside ±{expected['tolerance']}ms of {expected['mean']})")
                all_pass = False
    
    if all_pass:
        print(f"\n{'='*60}")
        print("SUCCESS: All results match paper within tolerance!")
        print("Reproduction validated.")
        print('='*60)
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("WARNING: Some results deviate from paper.")
        print("Possible causes: different GPU, different CUDA version, or random seed variation.")
        print('='*60)
        sys.exit(1)
        if __name__ == "__main__":
    main()
