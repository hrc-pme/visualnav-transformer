#!/usr/bin/env python3
"""
Copy data.yaml configuration to dataset directories.

This script copies the current data.yaml to all processed dataset directories
so each dataset can have its own configuration (especially metric_waypoint_spacing).
"""

import os
import sys
import shutil
import yaml

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
DATA_YAML_TEMPLATE = os.path.join(os.path.dirname(__file__), "data.yaml")


def copy_config_to_datasets():
    """Copy data.yaml to each dataset directory."""
    
    if not os.path.exists(DATASETS_DIR):
        print(f"❌ Datasets directory not found: {DATASETS_DIR}")
        return
    
    if not os.path.exists(DATA_YAML_TEMPLATE):
        print(f"❌ Template data.yaml not found: {DATA_YAML_TEMPLATE}")
        return
    
    # Find all dataset directories
    datasets = []
    for item in os.listdir(DATASETS_DIR):
        item_path = os.path.join(DATASETS_DIR, item)
        if os.path.isdir(item_path):
            datasets.append((item, item_path))
    
    if not datasets:
        print(f"⚠️  No datasets found in {DATASETS_DIR}")
        return
    
    print(f"\nFound {len(datasets)} dataset(s):\n")
    
    # Copy config to each dataset
    for dataset_name, dataset_path in datasets:
        config_dest = os.path.join(dataset_path, "data.yaml")
        
        # Check if config already exists
        if os.path.exists(config_dest):
            print(f"⏭️  {dataset_name}: config already exists, skipping")
            continue
        
        # Read template and update dataset name
        with open(DATA_YAML_TEMPLATE, 'r') as f:
            config = yaml.safe_load(f)
        
        config['dataset']['name'] = dataset_name
        
        # Write to dataset directory
        with open(config_dest, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ {dataset_name}: copied config")
    
    print(f"\n✅ Done! Each dataset now has its own data.yaml configuration.")
    print(f"   You can customize metric_waypoint_spacing for each dataset individually.")


if __name__ == "__main__":
    copy_config_to_datasets()
