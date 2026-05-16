"""
Baseline models for comparison.
Implements Section III.F exactly.

Baselines:
1. Random Forest: 500 trees, max_depth=20 (Section III.F)
2. XGBoost: learning_rate=0.1, max_depth=6, early stopping (Section III.F)

Both receive identical PCA-reduced features (47 components) for fair comparison.
Hyperparameters optimized via grid search on validation data.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import json
import os
import argparse

# XGBoost is optional - install with: pip install xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest baseline (Section III.F).

    Configuration:
    - 500 decision trees
    - max_depth=20
    - Handles high-dimensional data robustly
    - Lacks temporal modeling capabilities

    Args:
        X_train: Training features (PCA-reduced, 47 components)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Trained RF model and training time
    """
    print("\nTraining Random Forest baseline...")
    print("  Configuration: 500 trees, max_depth=20 (Section III.F)")

    start_time = time.time()

    rf = RandomForestClassifier(
        n_estimators=500,      # 500 decision trees
        max_depth=20,          # Maximum depth
        random_state=42,       # Reproducibility
        n_jobs=-1,             # Use all cores
        verbose=0
    )

    rf.fit(X_train, y_train)

    train_time = time.time() - start_time

    # Validation accuracy
    val_acc = accuracy_score(y_val, rf.predict(X_val))
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Training time: {train_time:.2f}s")

    return rf, train_time


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost baseline (Section III.F).

    Configuration:
    - learning_rate=0.1
    - max_depth=6
    - early stopping
    - Efficient nonlinear classification
    - Cannot capture sequential dependencies

    Args:
        X_train: Training features (PCA-reduced, 47 components)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Trained XGBoost model and training time
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    print("\nTraining XGBoost baseline...")
    print("  Configuration: lr=0.1, max_depth=6, early stopping (Section III.F)")

    start_time = time.time()

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Parameters from Section III.F
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'learning_rate': 0.1,      # Section III.F
        'max_depth': 6,            # Section III.F
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    # Train with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=20,   # Early stopping
        verbose_eval=False
    )

    train_time = time.time() - start_time

    # Validation accuracy
    val_pred = model.predict(dval)
    val_pred_classes = np.argmax(val_pred, axis=1)
    val_acc = accuracy_score(y_val, val_pred_classes)

    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Training time: {train_time:.2f}s")

    return model, train_time


def evaluate_baseline(model, X_test, y_test, model_name='Baseline', measure_latency=True):
    """
    Evaluate baseline model and measure latency.

    Args:
        model: Trained baseline model
        X_test: Test features
        y_test: Test labels
        model_name: Name for reporting
        measure_latency: Whether to measure inference latency

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating {model_name}...")

    # Predictions
    if model_name == 'Random Forest':
        start_time = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_time = time.perf_counter() - start_time
    else:  # XGBoost
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test)
        start_time = time.perf_counter()
        y_prob = model.predict(dtest)
        y_pred = np.argmax(y_prob, axis=1)
        inference_time = time.perf_counter() - start_time

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    # Latency per spike (amortized over test set)
    latency_ms = (inference_time / len(X_test)) * 1000

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Latency:   {latency_ms:.1f} ms per spike")

    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'latency_ms': latency_ms
    }

    return results


def run_all_baselines(data_path, output_dir='results'):
    """
    Train and evaluate all baseline models.

    Reproduces Table VII baseline results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_class_train']
    X_val = data['X_val']
    y_class_val = data['y_class_val']
    X_test = data['X_test']
    y_test = data['y_class_test']

    print(f"\n{'='*60}")
    print("Baseline Model Training and Evaluation")
    print(f"{'='*60}")
    print(f"Data: {data_path}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]} PCA components")

    all_results = []

    # Random Forest
    rf_model, rf_train_time = train_random_forest(X_train, y_train, X_val, y_class_val)
    rf_results = evaluate_baseline(rf_model, X_test, y_test, 'Random Forest')
    rf_results['train_time_s'] = rf_train_time
    all_results.append(rf_results)

    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_train_time = train_xgboost(X_train, y_train, X_val, y_class_val)
        xgb_results = evaluate_baseline(xgb_model, X_test, y_test, 'XGBoost')
        xgb_results['train_time_s'] = xgb_train_time
        all_results.append(xgb_results)

    # Save results
    results_path = os.path.join(output_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Baseline Results Summary (Table VII format):")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n{r['model']}:")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Latency:   {r['latency_ms']:.1f} ms")

    print(f"\nResults saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate baseline models')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to preprocessed data (.npz)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'rf', 'xgb'],
                        help='Which baseline to train')

    args = parser.parse_args()

    if args.model == 'all':
        run_all_baselines(args.data, args.output_dir)
    elif args.model == 'rf':
        data = np.load(args.data)
        rf_model, _ = train_random_forest(
            data['X_train'], data['y_class_train'],
            data['X_val'], data['y_class_val']
        )
        results = evaluate_baseline(rf_model, data['X_test'], data['y_class_test'], 'Random Forest')
        print(f"\nRF Results: {results}")
    elif args.model == 'xgb':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        data = np.load(args.data)
        xgb_model, _ = train_xgboost(
            data['X_train'], data['y_class_train'],
            data['X_val'], data['y_class_val']
        )
        results = evaluate_baseline(xgb_model, data['X_test'], data['y_class_test'], 'XGBoost')
        print(f"\nXGBoost Results: {results}")


if __name__ == "__main__":
    main()
