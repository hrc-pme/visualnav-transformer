#!/usr/bin/env python3
"""
Merge Multiple Datasets into One

This script merges multiple datasets from the datasets/ directory into a single
unified dataset. All trajectories are renumbered sequentially (0001, 0002, etc.).

Configuration:
- MERGE_DATASET_NAME: Name for the merged dataset
- SAVE_RAW: If True, also merge raw rosbag files
"""

import os
import sys
import shutil
import pickle
from pathlib import Path
from PIL import Image

# ============================================================================
# CONFIGURATION - Modify these variables
# ============================================================================

SAVE_RAW = True
MERGE_DATASET_NAME = 'stairs_1elevator'

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Get the directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, "..")
DATASETS_ROOT_DIR = os.path.join(TRAIN_DIR, "datasets")
DATA_SPLITS_DIR = os.path.join(TRAIN_DIR, "vint_train/data/data_splits")


def get_all_datasets():
    """Get list of all datasets (both processed and raw)."""
    datasets = []
    
    if not os.path.exists(DATASETS_ROOT_DIR):
        print(f"❌ Error: Datasets directory not found: {DATASETS_ROOT_DIR}")
        return datasets
    
    for item in os.listdir(DATASETS_ROOT_DIR):
        item_path = os.path.join(DATASETS_ROOT_DIR, item)
        if os.path.isdir(item_path):
            processed_path = os.path.join(item_path, "processed_data", item)
            rosbags_path = os.path.join(item_path, "rosbags")
            
            has_processed = os.path.exists(processed_path)
            has_rosbags = os.path.exists(rosbags_path)
            
            # Skip if neither processed data nor rosbags exist
            if not has_processed and not has_rosbags:
                continue
            
            num_trajs = 0
            num_bags = 0
            
            # Count trajectories if processed
            if has_processed:
                traj_dirs = [d for d in os.listdir(processed_path) 
                            if os.path.isdir(os.path.join(processed_path, d))]
                num_trajs = len(traj_dirs)
            
            # Count rosbags
            if has_rosbags:
                bag_dirs = [d for d in os.listdir(rosbags_path)
                           if os.path.isdir(os.path.join(rosbags_path, d))
                           and os.path.exists(os.path.join(rosbags_path, d, "metadata.yaml"))]
                num_bags = len(bag_dirs)
            
            # Add dataset if it has either processed data or rosbags
            if num_trajs > 0 or num_bags > 0:
                datasets.append({
                    'name': item,
                    'path': item_path,
                    'processed_path': processed_path if has_processed else None,
                    'rosbags_path': rosbags_path if has_rosbags else None,
                    'num_trajs': num_trajs,
                    'num_bags': num_bags,
                    'has_processed': has_processed,
                    'has_rosbags': has_rosbags
                })
    
    return sorted(datasets, key=lambda x: x['name'])


def merge_datasets():
    """Merge all datasets into one unified dataset."""
    
    print("\n" + "="*80)
    print("MERGE DATASETS")
    print("="*80)
    print(f"Target merged dataset name: {MERGE_DATASET_NAME}")
    print(f"Save raw rosbags: {SAVE_RAW}")
    print("="*80 + "\n")
    
    # Get all available datasets
    datasets = get_all_datasets()
    
    if len(datasets) == 0:
        print("❌ No datasets found!")
        return
    
    print(f"Found {len(datasets)} dataset(s) to merge:\n")
    for ds in datasets:
        status = []
        if ds['has_processed']:
            status.append(f"{ds['num_trajs']:3d} trajs")
        if ds['has_rosbags']:
            status.append(f"{ds['num_bags']:3d} bags")
        print(f"  - {ds['name']:30s} [{', '.join(status)}]")
    
    # Confirm merge
    print(f"\n⚠️  This will create a new merged dataset: {MERGE_DATASET_NAME}")
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Merge cancelled.")
        return
    
    # Create merged dataset directory structure
    merged_dataset_path = os.path.join(DATASETS_ROOT_DIR, MERGE_DATASET_NAME)
    merged_processed_path = os.path.join(merged_dataset_path, "processed_data", MERGE_DATASET_NAME)
    merged_rosbags_path = os.path.join(merged_dataset_path, "rosbags")
    
    if os.path.exists(merged_dataset_path):
        print(f"\n⚠️  Warning: Merged dataset directory already exists: {merged_dataset_path}")
        response = input("Remove existing and continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Merge cancelled.")
            return
        shutil.rmtree(merged_dataset_path)
        print(f"✓ Removed existing directory")
    
    os.makedirs(merged_processed_path, exist_ok=True)
    if SAVE_RAW:
        os.makedirs(merged_rosbags_path, exist_ok=True)
    
    print(f"\n📁 Created merged dataset directory: {merged_dataset_path}")
    
    # Merge all trajectories
    print(f"\n🔄 Merging trajectories...\n")
    
    trajectory_counter = 1
    source_mapping = []  # Track which source each trajectory came from
    
    for ds in datasets:
        print(f"Processing {ds['name']}...")
        
        # Skip datasets without processed data
        if not ds['has_processed']:
            print(f"  ⚠️  Skipping (no processed data) - run format_dataset.py first!")
            continue
        
        # Get all trajectory directories
        traj_dirs = sorted([d for d in os.listdir(ds['processed_path']) 
                           if os.path.isdir(os.path.join(ds['processed_path'], d))])
        
        for traj_dir in traj_dirs:
            src_traj_path = os.path.join(ds['processed_path'], traj_dir)
            
            # Create new trajectory name with zero-padded numbering
            new_traj_name = f"{trajectory_counter:04d}"
            dst_traj_path = os.path.join(merged_processed_path, new_traj_name)
            
            # Copy trajectory directory
            shutil.copytree(src_traj_path, dst_traj_path)
            
            # Track source mapping
            source_mapping.append({
                'merged_id': new_traj_name,
                'source_dataset': ds['name'],
                'source_traj': traj_dir
            })
            
            trajectory_counter += 1
        
        print(f"  ✓ Copied {len(traj_dirs)} trajectories from {ds['name']}")
        
        # Copy rosbags if SAVE_RAW is True
        if SAVE_RAW and ds['has_rosbags']:
            rosbags_src = ds['rosbags_path']
            if os.path.exists(rosbags_src):
                bag_dirs = [d for d in os.listdir(rosbags_src) 
                           if os.path.isdir(os.path.join(rosbags_src, d))]
                
                for bag_dir in bag_dirs:
                    src_bag_path = os.path.join(rosbags_src, bag_dir)
                    # Keep original bag name but add source dataset prefix to avoid conflicts
                    dst_bag_name = f"{ds['name']}_{bag_dir}"
                    dst_bag_path = os.path.join(merged_rosbags_path, dst_bag_name)
                    
                    shutil.copytree(src_bag_path, dst_bag_path)
                
                print(f"  ✓ Copied {len(bag_dirs)} rosbag(s) from {ds['name']}")
    
    total_trajs = trajectory_counter - 1
    print(f"\n✅ Successfully merged {total_trajs} trajectories from {len(datasets)} datasets")
    
    # Save source mapping file
    mapping_file = os.path.join(merged_dataset_path, "source_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("# Source Mapping for Merged Dataset\n")
        f.write(f"# Merged Dataset: {MERGE_DATASET_NAME}\n")
        f.write("#\n")
        f.write("# Format: merged_traj_id | source_dataset | original_traj_name\n")
        f.write("#\n\n")
        for mapping in source_mapping:
            f.write(f"{mapping['merged_id']} | {mapping['source_dataset']} | {mapping['source_traj']}\n")
    
    print(f"📝 Saved source mapping to: {mapping_file}")
    
    # Merge train/test splits
    print(f"\n🔄 Creating merged train/test splits...\n")
    
    merged_splits_dir = os.path.join(DATA_SPLITS_DIR, MERGE_DATASET_NAME)
    os.makedirs(os.path.join(merged_splits_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(merged_splits_dir, "test"), exist_ok=True)
    
    train_trajs = []
    test_trajs = []
    
    for ds in datasets:
        # Skip datasets without processed data (already skipped in merging above)
        if not ds['has_processed']:
            continue
            
        # Check if splits exist for this dataset
        ds_splits_dir = os.path.join(DATA_SPLITS_DIR, ds['name'])
        train_split_file = os.path.join(ds_splits_dir, "train", "traj_names.txt")
        test_split_file = os.path.join(ds_splits_dir, "test", "traj_names.txt")
        
        if os.path.exists(train_split_file):
            with open(train_split_file, 'r') as f:
                ds_train = [line.strip() for line in f if line.strip()]
        else:
            ds_train = []
        
        if os.path.exists(test_split_file):
            with open(test_split_file, 'r') as f:
                ds_test = [line.strip() for line in f if line.strip()]
        else:
            ds_test = []
        
        # Map old trajectory names to new merged names
        for mapping in source_mapping:
            if mapping['source_dataset'] == ds['name']:
                old_name = mapping['source_traj']
                new_name = mapping['merged_id']
                
                if old_name in ds_train:
                    train_trajs.append(new_name)
                elif old_name in ds_test:
                    test_trajs.append(new_name)
    
    # Write merged split files
    with open(os.path.join(merged_splits_dir, "train", "traj_names.txt"), 'w') as f:
        for traj in sorted(train_trajs):
            f.write(traj + "\n")
    
    with open(os.path.join(merged_splits_dir, "test", "traj_names.txt"), 'w') as f:
        for traj in sorted(test_trajs):
            f.write(traj + "\n")
    
    print(f"  ✓ Train split: {len(train_trajs)} trajectories")
    print(f"  ✓ Test split:  {len(test_trajs)} trajectories")
    print(f"  ✓ Saved to: {merged_splits_dir}")
    
    # Copy data.yaml if it exists in any source dataset (prioritize processed datasets)
    for ds in datasets:
        if not ds['has_processed']:
            continue
        data_yaml_src = os.path.join(ds['path'], "data.yaml")
        if os.path.exists(data_yaml_src):
            data_yaml_dst = os.path.join(merged_dataset_path, "data.yaml")
            shutil.copy2(data_yaml_src, data_yaml_dst)
            
            # Update dataset name in the copied config
            import yaml
            with open(data_yaml_dst, 'r') as f:
                config = yaml.safe_load(f)
            
            config['dataset']['name'] = MERGE_DATASET_NAME
            config['dataset']['description'] = f"Merged dataset from: {', '.join([d['name'] for d in datasets])}"
            
            with open(data_yaml_dst, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"\n📝 Copied and updated data.yaml configuration")
            break
    
    # Summary
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"Merged dataset name: {MERGE_DATASET_NAME}")
    print(f"Location: {merged_dataset_path}")
    print(f"Total trajectories: {total_trajs}")
    print(f"  - Train: {len(train_trajs)}")
    print(f"  - Test:  {len(test_trajs)}")
    if SAVE_RAW:
        bag_count = len([d for d in os.listdir(merged_rosbags_path) 
                        if os.path.isdir(os.path.join(merged_rosbags_path, d))])
        print(f"Rosbags: {bag_count}")
    print("\n📋 Next steps:")
    print(f"   1. The merged dataset is ready to use")
    print(f"   2. Update vint_train/data/data_config.yaml to include:")
    print(f"      {MERGE_DATASET_NAME}:")
    print(f"        metric_waypoint_spacing: 0.25")
    print(f"   3. Run training with the merged dataset")
    print("="*80 + "\n")


def main():
    """Main function."""
    try:
        merge_datasets()
    except KeyboardInterrupt:
        print("\n\n❌ Merge cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
