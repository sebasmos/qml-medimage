#!/usr/bin/env python3
"""
Automated PCA Reduction Pipeline
=================================

Processes all pkl files in a directory one by one:
1. Applies PCA reduction to each file sequentially
2. Verifies the output format matches the original
3. Generates a CSV summary of all processing

Key Features:
  - Processes files ONE AT A TIME (memory safe)
  - Verifies each output before moving to next
  - Creates detailed CSV summary
  - Supports resuming (skips already processed files)
  - Logs all operations

Usage:
    # Process all files with default settings
    python run_pca_pipeline.py
    
    # Process specific range of files
    python run_pca_pipeline.py --start 1 --end 5
    
    # Force reprocess (don't skip existing)
    python run_pca_pipeline.py --force
    
    # Custom n_components
    python run_pca_pipeline.py --n_components 500

Output:
    - PCA reduced pkl files in data-cleaned-pca-{n_components}/
    - Summary CSV: pca_processing_summary.csv
    - Log file: pca_processing.log
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import csv
import gc

import numpy as np
import pandas as pd


def get_available_files(input_dir: str) -> List[int]:
    """Get list of available file numbers in the input directory."""
    input_path = Path(input_dir)
    files = list(input_path.glob("data_type*_insurance.pkl"))
    
    file_nums = []
    for f in files:
        # Extract number from data_type{N}_insurance.pkl
        name = f.stem  # data_type{N}_insurance
        try:
            num = int(name.split('type')[1].split('_')[0])
            file_nums.append(num)
        except (IndexError, ValueError):
            continue
    
    return sorted(file_nums)


def check_output_exists(output_dir: str, file_num: int) -> bool:
    """Check if output file already exists."""
    output_path = Path(output_dir) / f"data_type{file_num}_insurance.pkl"
    return output_path.exists()


def get_file_info(filepath: Path) -> Dict:
    """Get basic info about a pkl file without loading all data."""
    df = pd.read_pickle(filepath)
    
    info = {
        "n_samples": len(df),
        "columns": list(df.columns),
        "embedding_shape": df["embedding"].iloc[0].shape if "embedding" in df.columns else None,
    }
    
    # Check for insurance column
    insurance_cols = [c for c in df.columns if 'insurance' in c.lower()]
    if insurance_cols:
        info["insurance_column"] = insurance_cols[0]
        info["insurance_unique_values"] = df[insurance_cols[0]].nunique()
    
    # Check for label column
    if "label" in df.columns:
        info["label_unique_values"] = df["label"].nunique()
    
    # Get PCA attributes if present
    if hasattr(df, 'attrs'):
        info["pca_n_components"] = df.attrs.get("pca_n_components")
        info["pca_variance_explained"] = df.attrs.get("pca_variance_explained")
        info["original_embedding_shape"] = df.attrs.get("original_embedding_shape")
    
    del df
    gc.collect()
    
    return info


def verify_pca_output(original_path: Path, pca_path: Path) -> Dict:
    """Verify PCA output matches original format."""
    result = {
        "verified": False,
        "errors": [],
        "warnings": []
    }
    
    if not original_path.exists():
        result["errors"].append("Original file not found")
        return result
    
    if not pca_path.exists():
        result["errors"].append("PCA file not found")
        return result
    
    try:
        df_orig = pd.read_pickle(original_path)
        df_pca = pd.read_pickle(pca_path)
        
        # Check sample count
        if len(df_orig) != len(df_pca):
            result["errors"].append(f"Sample count mismatch: {len(df_orig)} vs {len(df_pca)}")
        
        # Check columns
        if set(df_orig.columns) != set(df_pca.columns):
            result["errors"].append("Column mismatch")
        
        # Check insurance labels preserved
        insurance_cols = [c for c in df_orig.columns if 'insurance' in c.lower()]
        for col in insurance_cols:
            if col in df_pca.columns:
                if not df_orig[col].equals(df_pca[col]):
                    # Handle NaN comparison
                    orig_vals = df_orig[col].fillna("__NAN__")
                    pca_vals = df_pca[col].fillna("__NAN__")
                    if not orig_vals.equals(pca_vals):
                        result["errors"].append(f"Insurance column '{col}' values differ")
        
        # Check label column preserved
        if "label" in df_orig.columns and "label" in df_pca.columns:
            if not df_orig["label"].equals(df_pca["label"]):
                result["errors"].append("Label column values differ")
        
        # Check embedding reduced
        orig_emb_size = np.prod(df_orig["embedding"].iloc[0].shape)
        pca_emb_size = np.prod(df_pca["embedding"].iloc[0].shape)
        
        if pca_emb_size >= orig_emb_size:
            result["warnings"].append("Embedding not reduced")
        
        # Check PCA metadata
        if not hasattr(df_pca, 'attrs') or "pca_n_components" not in df_pca.attrs:
            result["warnings"].append("Missing PCA metadata")
        
        if not result["errors"]:
            result["verified"] = True
        
        del df_orig, df_pca
        gc.collect()
        
    except Exception as e:
        result["errors"].append(f"Verification error: {str(e)}")
    
    return result


def run_pca_reduction(
    file_num: int,
    input_dir: str,
    output_base: str,
    n_components: int,
    batch_size: int = 200,
    script_path: str = None
) -> Dict:
    """Run PCA reduction on a single file."""
    
    result = {
        "file_num": file_num,
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "error_message": None
    }
    
    # Determine script path
    if script_path is None:
        script_path = Path(__file__).parent / "apply_pca_reduction.py"
    
    if not Path(script_path).exists():
        result["status"] = "failed"
        result["error_message"] = f"PCA script not found: {script_path}"
        return result
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--file_num", str(file_num),
        "--n_components", str(n_components),
        "--input_dir", input_dir,
        "--output_base", output_base,
        "--batch_size", str(batch_size)
    ]
    
    print(f"\n{'='*60}")
    print(f"Processing file_num={file_num}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = datetime.now()
    
    try:
        # Run the PCA script
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration_seconds"] = (end_time - start_time).total_seconds()
        
        if process.returncode == 0:
            result["status"] = "success"
            print(f"  SUCCESS (took {result['duration_seconds']:.1f}s)")
        else:
            result["status"] = "failed"
            result["error_message"] = process.stderr[:500] if process.stderr else "Unknown error"
            print(f"  FAILED: {result['error_message']}")
        
        # Print stdout for debugging
        if process.stdout:
            print(process.stdout[-2000:])  # Last 2000 chars
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error_message"] = "Process timed out after 1 hour"
        print(f"  TIMEOUT")
    except Exception as e:
        result["status"] = "error"
        result["error_message"] = str(e)
        print(f"  ERROR: {e}")
    
    return result


def create_summary_csv(
    summary_data: List[Dict],
    output_path: Path
):
    """Create CSV summary of all processing."""
    
    if not summary_data:
        print("No data to write to CSV")
        return
    
    # Get all unique keys
    all_keys = set()
    for row in summary_data:
        all_keys.update(row.keys())
    
    # Sort keys for consistent column order
    fieldnames = sorted(all_keys)
    
    # Move important columns to front
    priority_cols = [
        "file_num", "status", "verified", "n_samples", 
        "original_features", "pca_features", "variance_explained",
        "duration_seconds", "error_message"
    ]
    
    ordered_fieldnames = []
    for col in priority_cols:
        if col in fieldnames:
            ordered_fieldnames.append(col)
            fieldnames.remove(col)
    ordered_fieldnames.extend(fieldnames)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"\nSummary CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated PCA reduction pipeline for all pkl files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/DATASET/data-cleaned",
        help="Input directory containing pickle files"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="/orcd/pool/006/lceli_shared/DATASET",
        help="Base output directory"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=1999,
        help="Number of PCA components"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start file number (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End file number (inclusive)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess even if output exists"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing outputs, don't process"
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="pca_processing_summary.csv",
        help="Output summary CSV filename"
    )
    parser.add_argument(
        "--pca_script",
        type=str,
        default=None,
        help="Path to apply_pca_reduction.py script"
    )

    args = parser.parse_args()

    # Construct output directory
    output_dir = os.path.join(args.output_base, f"data-cleaned-pca-{args.n_components}")

    print("=" * 60)
    print("AUTOMATED PCA REDUCTION PIPELINE")
    print("=" * 60)
    print(f"\nInput directory:  {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"N components:     {args.n_components}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Force reprocess:  {args.force}")
    print(f"Verify only:      {args.verify_only}")

    # Get available files
    available_files = get_available_files(args.input_dir)
    
    if not available_files:
        print(f"\nERROR: No pkl files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(available_files)} files: {available_files}")

    # Filter by range if specified
    if args.start is not None:
        available_files = [f for f in available_files if f >= args.start]
    if args.end is not None:
        available_files = [f for f in available_files if f <= args.end]
    
    print(f"Processing files: {available_files}")

    # Process each file
    summary_data = []
    
    for file_num in available_files:
        row = {
            "file_num": file_num,
            "input_file": f"data_type{file_num}_insurance.pkl",
            "timestamp": datetime.now().isoformat()
        }
        
        original_path = Path(args.input_dir) / f"data_type{file_num}_insurance.pkl"
        pca_path = Path(output_dir) / f"data_type{file_num}_insurance.pkl"
        
        # Get original file info
        print(f"\n--- File {file_num} ---")
        print(f"Getting info from: {original_path.name}")
        
        try:
            orig_info = get_file_info(original_path)
            row["n_samples"] = orig_info["n_samples"]
            row["original_features"] = np.prod(orig_info["embedding_shape"]) if orig_info["embedding_shape"] else None
            row["original_embedding_shape"] = str(orig_info["embedding_shape"])
            row["columns"] = str(orig_info["columns"])
            row["insurance_column"] = orig_info.get("insurance_column")
            row["insurance_unique_values"] = orig_info.get("insurance_unique_values")
            row["label_unique_values"] = orig_info.get("label_unique_values")
        except Exception as e:
            row["error_message"] = f"Failed to read original: {e}"
            row["status"] = "failed"
            summary_data.append(row)
            continue
        
        # Check if output exists
        output_exists = check_output_exists(output_dir, file_num)
        row["output_existed"] = output_exists
        
        # Skip or process
        if args.verify_only:
            if not output_exists:
                row["status"] = "skipped"
                row["verified"] = False
                print(f"  Output not found, skipping verification")
                summary_data.append(row)
                continue
        elif output_exists and not args.force:
            print(f"  Output already exists, skipping (use --force to reprocess)")
            row["status"] = "skipped_exists"
        else:
            # Run PCA reduction
            pca_result = run_pca_reduction(
                file_num=file_num,
                input_dir=args.input_dir,
                output_base=args.output_base,
                n_components=args.n_components,
                batch_size=args.batch_size,
                script_path=args.pca_script
            )
            
            row["status"] = pca_result["status"]
            row["duration_seconds"] = pca_result.get("duration_seconds")
            row["error_message"] = pca_result.get("error_message")
            
            if pca_result["status"] != "success":
                summary_data.append(row)
                continue
        
        # Verify output
        print(f"  Verifying output...")
        verify_result = verify_pca_output(original_path, pca_path)
        row["verified"] = verify_result["verified"]
        row["verification_errors"] = str(verify_result["errors"]) if verify_result["errors"] else None
        row["verification_warnings"] = str(verify_result["warnings"]) if verify_result["warnings"] else None
        
        if verify_result["verified"]:
            print(f"  Verification PASSED")
        else:
            print(f"  Verification FAILED: {verify_result['errors']}")
        
        # Get PCA output info
        if pca_path.exists():
            try:
                pca_info = get_file_info(pca_path)
                row["pca_features"] = np.prod(pca_info["embedding_shape"]) if pca_info["embedding_shape"] else None
                row["pca_embedding_shape"] = str(pca_info["embedding_shape"])
                row["pca_n_components"] = pca_info.get("pca_n_components")
                row["variance_explained"] = pca_info.get("pca_variance_explained")
                
                if row["variance_explained"]:
                    row["variance_explained_pct"] = f"{row['variance_explained']*100:.2f}%"
            except Exception as e:
                row["pca_info_error"] = str(e)
        
        summary_data.append(row)
        
        # Force garbage collection between files
        gc.collect()

    # Create summary CSV
    summary_path = Path(output_dir) / args.summary_file
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    create_summary_csv(summary_data, summary_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    total = len(summary_data)
    success = sum(1 for r in summary_data if r.get("status") == "success")
    skipped = sum(1 for r in summary_data if r.get("status", "").startswith("skipped"))
    failed = sum(1 for r in summary_data if r.get("status") == "failed")
    verified = sum(1 for r in summary_data if r.get("verified") == True)
    
    print(f"\nTotal files:    {total}")
    print(f"Success:        {success}")
    print(f"Skipped:        {skipped}")
    print(f"Failed:         {failed}")
    print(f"Verified:       {verified}/{total}")
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Return exit code based on results
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())