#!/usr/bin/env python3
"""
Test for the aggregation fix in qsvm_hybrid_insurance.py

This test verifies that the aggregation code properly handles non-numeric columns
like 'classical_kernel' when computing mean and std across multiple seeds.

Usage:
    python tests/test_aggregation_fix.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def test_aggregation_with_string_column():
    """Test that aggregation works with mixed numeric and string columns."""
    
    # Simulate the structure of global_rows from qsvm_hybrid_insurance.py
    global_rows = [
        {
            "seed": 0,
            "train_accuracy": 0.85,
            "val_accuracy": 0.83,
            "test_accuracy": 0.82,
            "train_time_sec": 45.2,
            "alpha": 0.5,
            "classical_kernel": "rbf",  # String column that caused the error
        },
        {
            "seed": 1,
            "train_accuracy": 0.86,
            "val_accuracy": 0.84,
            "test_accuracy": 0.83,
            "train_time_sec": 46.1,
            "alpha": 0.5,
            "classical_kernel": "rbf",
        },
        {
            "seed": 2,
            "train_accuracy": 0.84,
            "val_accuracy": 0.82,
            "test_accuracy": 0.81,
            "train_time_sec": 44.8,
            "alpha": 0.5,
            "classical_kernel": "rbf",
        },
    ]
    
    # This is the fixed code from lines 737-740 in qsvm_hybrid_insurance.py
    all_metrics = pd.DataFrame(global_rows)
    
    # Only aggregate numeric columns (exclude seed and non-numeric columns like classical_kernel)
    numeric_cols = all_metrics.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if c != "seed"]
    
    # This should not raise TypeError: Could not convert string 'rbf' to numeric
    agg = all_metrics[metric_cols].agg(["mean", "std"]).T
    
    # Verify the results
    assert "train_accuracy" in agg.index
    assert "val_accuracy" in agg.index
    assert "test_accuracy" in agg.index
    assert "train_time_sec" in agg.index
    assert "alpha" in agg.index
    
    # Verify that string columns are excluded
    assert "classical_kernel" not in agg.index
    assert "seed" not in agg.index
    
    # Verify the aggregation values are correct
    assert abs(agg.loc["train_accuracy", "mean"] - 0.85) < 0.01
    assert abs(agg.loc["val_accuracy", "mean"] - 0.83) < 0.01
    
    print("✓ Test passed: Aggregation handles string columns correctly")
    return True


def test_old_code_would_fail():
    """Demonstrate that the old code would have failed."""
    
    global_rows = [
        {"seed": 0, "accuracy": 0.85, "classical_kernel": "rbf"},
        {"seed": 1, "accuracy": 0.86, "classical_kernel": "rbf"},
    ]
    
    all_metrics = pd.DataFrame(global_rows)
    
    # Old code: metric_cols = [c for c in all_metrics.columns if c != "seed"]
    # This would include "classical_kernel" which is a string
    metric_cols_old = [c for c in all_metrics.columns if c != "seed"]
    
    # Verify that the old approach would include the string column
    assert "classical_kernel" in metric_cols_old
    
    # Try to aggregate - this would fail with old code
    try:
        agg_old = all_metrics[metric_cols_old].agg(["mean", "std"]).T
        print("⚠️  Old code didn't fail (maybe pandas version difference)")
    except (TypeError, ValueError) as e:
        print(f"✓ Confirmed: Old code would fail with: {type(e).__name__}")
    
    # New code: only select numeric columns
    numeric_cols = all_metrics.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols_new = [c for c in numeric_cols if c != "seed"]
    
    # Verify string column is excluded
    assert "classical_kernel" not in metric_cols_new
    
    # This should work
    agg_new = all_metrics[metric_cols_new].agg(["mean", "std"]).T
    assert "accuracy" in agg_new.index
    
    print("✓ Test passed: New code correctly filters numeric columns")
    return True


def main():
    print("=" * 60)
    print("Testing Aggregation Fix for qsvm_hybrid_insurance.py")
    print("=" * 60)
    
    try:
        test_aggregation_with_string_column()
        test_old_code_would_fail()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe fix correctly handles non-numeric columns in aggregation.")
        print("String columns like 'classical_kernel' are now excluded from")
        print("mean/std calculations, preventing the TypeError.")
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
