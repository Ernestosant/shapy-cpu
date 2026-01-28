#!/usr/bin/env python3
"""
SHAPY Data Setup Script

Reorganizes SHAPY_FILES folder to match the expected structure for SHAPY inference.
Creates symbolic links (or copies) to organize the data properly.

Usage:
    python setup_data.py --source SHAPY_FILES --target data
    python setup_data.py --source SHAPY_FILES --target data --copy  # Copy instead of symlink
"""

import os
import os.path as osp
import shutil
import argparse
from pathlib import Path


def create_link_or_copy(source: Path, target: Path, copy: bool = False):
    """Create symbolic link or copy file/directory."""
    if target.exists():
        print(f"  [SKIP] Already exists: {target}")
        return
    
    target.parent.mkdir(parents=True, exist_ok=True)
    
    if copy:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        print(f"  [COPY] {source} -> {target}")
    else:
        try:
            target.symlink_to(source.resolve())
            print(f"  [LINK] {source} -> {target}")
        except OSError as e:
            # Windows may require admin for symlinks
            print(f"  [WARN] Symlink failed, copying instead: {e}")
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            print(f"  [COPY] {source} -> {target}")


def setup_data(source_folder: str, target_folder: str, copy: bool = False):
    """
    Setup data folder structure from SHAPY_FILES.
    
    Expected source structure (SHAPY_FILES):
        - neutral/model.npz
        - female/model.npz
        - male/model.npz
        - expose_release/data/all_means.pkl
        - expose_release/data/SMPLX_to_J14.pkl
        - expose_release/utility_files/flame/
        - trained_models/shapy/SHAPY_A/
        - trained_models/b2a/polynomial/
        - utility_files/measurements/
        - utility_files/smplx/
        - utility_files/shape_priors/
    
    Target structure (data/):
        - body_models/smplx/SMPLX_NEUTRAL.npz
        - body_models/smplx/SMPLX_FEMALE.npz
        - body_models/smplx/SMPLX_MALE.npz
        - expose_release/data/...
        - trained_models/...
        - utility_files/...
    """
    source = Path(source_folder).resolve()
    target = Path(target_folder).resolve()
    
    print(f"Setting up data structure...")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Mode: {'copy' if copy else 'symlink'}")
    print()
    
    if not source.exists():
        print(f"ERROR: Source folder not found: {source}")
        return False
    
    # Create target folder
    target.mkdir(parents=True, exist_ok=True)
    
    # Mapping: (source_path, target_path)
    mappings = []
    
    # ==== Body Models ====
    print("Setting up body_models...")
    smplx_target = target / 'body_models' / 'smplx'
    smplx_target.mkdir(parents=True, exist_ok=True)
    
    # SMPL-X models
    gender_mapping = {
        'neutral': 'SMPLX_NEUTRAL.npz',
        'female': 'SMPLX_FEMALE.npz',
        'male': 'SMPLX_MALE.npz',
    }
    
    for gender, target_name in gender_mapping.items():
        src = source / gender / 'model.npz'
        tgt = smplx_target / target_name
        if src.exists():
            create_link_or_copy(src, tgt, copy)
        else:
            print(f"  [WARN] Not found: {src}")
    
    # ==== ExPose Release ====
    print("\nSetting up expose_release...")
    expose_src = source / 'expose_release'
    expose_tgt = target / 'expose_release'
    
    if expose_src.exists():
        # Data files
        for name in ['all_means.pkl', 'SMPLX_to_J14.pkl']:
            src = expose_src / 'data' / name
            tgt = expose_tgt / 'data' / name
            if src.exists():
                create_link_or_copy(src, tgt, copy)
        
        # Utility files
        flame_src = expose_src / 'utility_files' / 'flame'
        flame_tgt = expose_tgt / 'utility_files' / 'flame'
        if flame_src.exists():
            create_link_or_copy(flame_src, flame_tgt, copy)
    
    # ==== Trained Models ====
    print("\nSetting up trained_models...")
    
    # SHAPY_A
    shapy_src = source / 'trained_models' / 'shapy'
    shapy_tgt = target / 'trained_models' / 'shapy'
    if shapy_src.exists():
        create_link_or_copy(shapy_src, shapy_tgt, copy)
    
    # B2A models
    b2a_src = source / 'trained_models' / 'b2a'
    b2a_tgt = target / 'trained_models' / 'b2a'
    if b2a_src.exists():
        create_link_or_copy(b2a_src, b2a_tgt, copy)
    
    # A2B models (if exist)
    a2b_src = source / 'trained_models' / 'a2b'
    a2b_tgt = target / 'trained_models' / 'a2b'
    if a2b_src.exists():
        create_link_or_copy(a2b_src, a2b_tgt, copy)
    
    # ==== Utility Files ====
    print("\nSetting up utility_files...")
    util_src = source / 'utility_files'
    util_tgt = target / 'utility_files'
    
    if util_src.exists():
        # Measurements
        meas_src = util_src / 'measurements'
        meas_tgt = util_tgt / 'measurements'
        if meas_src.exists():
            create_link_or_copy(meas_src, meas_tgt, copy)
        
        # SMPLX utilities
        smplx_util_src = util_src / 'smplx'
        smplx_util_tgt = util_tgt / 'smplx'
        if smplx_util_src.exists():
            create_link_or_copy(smplx_util_src, smplx_util_tgt, copy)
        
        # Shape priors
        priors_src = util_src / 'shape_priors'
        priors_tgt = util_tgt / 'shape_priors'
        if priors_src.exists():
            create_link_or_copy(priors_src, priors_tgt, copy)
        
        # Evaluation
        eval_src = util_src / 'evaluation'
        eval_tgt = util_tgt / 'evaluation'
        if eval_src.exists():
            create_link_or_copy(eval_src, eval_tgt, copy)
    
    print("\n" + "="*50)
    print("Data setup complete!")
    print(f"Target folder: {target}")
    print("="*50)
    
    # Verify structure
    print("\nVerifying structure...")
    required_files = [
        'body_models/smplx/SMPLX_NEUTRAL.npz',
        'expose_release/data/all_means.pkl',
        'expose_release/data/SMPLX_to_J14.pkl',
        'trained_models/shapy/SHAPY_A/checkpoints/best_checkpoint',
        'utility_files/measurements/measurement_defitions.yaml',
        'utility_files/measurements/smplx_measurements.yaml',
    ]
    
    all_ok = True
    for rel_path in required_files:
        full_path = target / rel_path
        if full_path.exists():
            print(f"  [OK] {rel_path}")
        else:
            print(f"  [MISSING] {rel_path}")
            all_ok = False
    
    if all_ok:
        print("\nAll required files present!")
    else:
        print("\nWARNING: Some required files are missing!")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Setup SHAPY data folder structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_data.py --source SHAPY_FILES --target data
  python setup_data.py --source SHAPY_FILES --target data --copy
        """
    )
    
    parser.add_argument('--source', '-s', type=str, default='SHAPY_FILES',
                        help='Source folder (SHAPY_FILES)')
    parser.add_argument('--target', '-t', type=str, default='data',
                        help='Target folder (data)')
    parser.add_argument('--copy', '-c', action='store_true',
                        help='Copy files instead of creating symlinks')
    
    args = parser.parse_args()
    
    success = setup_data(args.source, args.target, args.copy)
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main()
