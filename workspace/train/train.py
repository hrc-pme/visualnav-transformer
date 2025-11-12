import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb
import sys
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

# ============================================================================
# CONFIGURATION - Modify these constants as needed
# ============================================================================

# Config file path
CONFIG_FILE = "config/vint.yaml"

# W&B Entity (change this to your wandb entity)
# Set to None to use your default entity, or specify your wandb username/team
WANDB_ENTITY = None  # Will use your logged-in account's default entity

# Datasets directory
DATASETS_DIR = "datasets"

# ============================================================================
# END CONFIGURATION
# ============================================================================

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)


# ============================================================================
# Model Download Functions
# ============================================================================

# Foundation model download URLs
FOUNDATION_MODEL_URLS = {
    "gnm": "https://drive.google.com/file/d/1bzCPd_OsXjS2aGPTQladbI8ImxLZwrQh/view?usp=drive_link",
    "vint": "https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view?usp=drive_link", 
    "nomad": "https://drive.google.com/file/d/1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb/view?usp=drive_link"
}

def extract_google_drive_id(url):
    """從 Google Drive URL 提取檔案 ID"""
    if "drive.google.com" in url:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
    return None

def download_model_from_google_drive(file_id, destination):
    """從 Google Drive 下載模型檔案"""
    # 使用 gdown 下載 Google Drive 檔案
    try:
        import gdown
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"📥 Downloading model to {destination}...")
        gdown.download(download_url, destination, quiet=False)
        return True
    except ImportError:
        print("⚠️  gdown not found. Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            download_url = f"https://drive.google.com/uc?id={file_id}"
            print(f"📥 Downloading model to {destination}...")
            gdown.download(download_url, destination, quiet=False)
            return True
        except Exception as e:
            print(f"❌ Failed to install gdown or download file: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def ensure_foundation_model_exists(model_name, model_path):
    """檢查 foundation model 是否存在，如果不存在則自動下載"""
    if os.path.exists(model_path):
        print(f"✓ Foundation model {model_name} already exists at {model_path}")
        return True
    
    print(f"⚠️  Foundation model {model_name} not found at {model_path}")
    
    # 確保模型目錄存在
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    print(f"📁 Created directory: {model_dir}")
    
    if model_name in FOUNDATION_MODEL_URLS:
        file_id = extract_google_drive_id(FOUNDATION_MODEL_URLS[model_name])
        if file_id:
            print(f"🌐 Downloading {model_name.upper()} foundation model from Google Drive...")
            if download_model_from_google_drive(file_id, model_path):
                print(f"✅ Successfully downloaded {model_name.upper()} foundation model")
                return True
            else:
                print(f"❌ Failed to download {model_name.upper()} foundation model")
                return False
        else:
            print(f"❌ Invalid Google Drive URL for {model_name}")
            return False
    else:
        print(f"❌ No download URL configured for model: {model_name}")
        return False


def find_all_datasets(datasets_dir=DATASETS_DIR):
    """
    Find all valid datasets in the datasets directory.
    A valid dataset must have:
    1. A processed_data subdirectory
    2. At least one trajectory in processed_data/<dataset_name>/
    
    Returns:
        dict: {dataset_name: dataset_info_dict}
    """
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        print(f"⚠️  Datasets directory {datasets_dir} does not exist!")
        return {}
    
    valid_datasets = {}
    
    for item in datasets_path.iterdir():
        if not item.is_dir():
            continue
            
        dataset_name = item.name
        processed_data_path = item / "processed_data" / dataset_name
        
        # Check if processed_data exists and has content
        if not processed_data_path.exists():
            print(f"⚠️  Skipping {dataset_name}: no processed_data directory")
            continue
        
        # Count trajectories (subdirectories in processed_data)
        trajectories = [d for d in processed_data_path.iterdir() if d.is_dir()]
        
        if len(trajectories) == 0:
            print(f"⚠️  Skipping {dataset_name}: processed_data is empty")
            continue
        
        # Check if data splits exist
        train_split = Path(f"vint_train/data/data_splits/{dataset_name}/train/traj_names.txt")
        test_split = Path(f"vint_train/data/data_splits/{dataset_name}/test/traj_names.txt")
        
        if not (train_split.exists() and test_split.exists()):
            print(f"⚠️  Skipping {dataset_name}: data splits not found")
            continue
        
        valid_datasets[dataset_name] = {
            'processed_data_path': str(processed_data_path),
            'train_split': str(train_split.parent),
            'test_split': str(test_split.parent),
            'num_trajectories': len(trajectories)
        }
        
        print(f"✅ Found valid dataset: {dataset_name} ({len(trajectories)} trajectories)")
    
    return valid_datasets


def build_datasets_config(valid_datasets, base_config):
    """
    Build the datasets configuration from discovered datasets.
    
    Args:
        valid_datasets: dict from find_all_datasets()
        base_config: base configuration dict
    
    Returns:
        dict: datasets configuration for the config
    """
    datasets_config = {}
    
    # Get default parameters
    default_params = base_config.get('default_dataset_params', {})
    
    for dataset_name, info in valid_datasets.items():
        datasets_config[dataset_name] = {
            'data_folder': os.path.abspath(info['processed_data_path']),
            'train': os.path.abspath(info['train_split']) + '/',
            'test': os.path.abspath(info['test_split']) + '/',
            'end_slack': default_params.get('end_slack', 0),
            'goals_per_obs': default_params.get('goals_per_obs', 3),
            'negative_mining': default_params.get('negative_mining', True),
        }
    
    return datasets_config


def ensure_datasets_in_config(dataset_names, datasets_dir=DATASETS_DIR, default_waypoint_spacing=0.25):
    """
    Ensure all discovered datasets are in data_config.yaml.
    Reads metric_waypoint_spacing from each dataset's data.yaml if available.
    
    Args:
        dataset_names: List of dataset names
        datasets_dir: Directory containing datasets
        default_waypoint_spacing: Default metric waypoint spacing for new datasets
    """
    data_config_path = os.path.join("vint_train", "data", "data_config.yaml")
    
    # Load existing config
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Check which datasets are missing
    modified = False
    for dataset_name in dataset_names:
        if dataset_name not in data_config:
            # Try to read metric_waypoint_spacing from dataset's data.yaml
            waypoint_spacing = default_waypoint_spacing
            dataset_config_path = os.path.join(datasets_dir, dataset_name, "data.yaml")
            if os.path.exists(dataset_config_path):
                try:
                    with open(dataset_config_path, 'r') as f:
                        dataset_config = yaml.safe_load(f)
                        waypoint_spacing = dataset_config.get('training', {}).get('metric_waypoint_spacing', default_waypoint_spacing)
                except Exception as e:
                    print(f"⚠️  Could not read {dataset_config_path}: {e}")
            
            print(f"📝 Adding {dataset_name} to data_config.yaml (waypoint_spacing: {waypoint_spacing}m)")
            data_config[dataset_name] = {
                'metric_waypoint_spacing': waypoint_spacing
            }
            modified = True
    
    # Save if modified
    if modified:
        with open(data_config_path, 'w') as f:
            yaml.safe_dump(data_config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Updated {data_config_path}\n")
    
    return modified


def main(config):
    # Auto-discover all datasets
    print("\n" + "="*80)
    print("SCANNING FOR DATASETS")
    print("="*80)
    
    valid_datasets = find_all_datasets(DATASETS_DIR)
    
    if not valid_datasets:
        print("\n❌ No valid datasets found! Please add datasets to the datasets/ directory.")
        print("   Each dataset should have:")
        print("   - datasets/<name>/processed_data/<name>/  (with trajectory folders)")
        print("   - vint_train/data/data_splits/<name>/train/traj_names.txt")
        print("   - vint_train/data/data_splits/<name>/test/traj_names.txt")
        return
    
    print(f"\n✅ Found {len(valid_datasets)} valid dataset(s)")
    print("="*80 + "\n")
    
    # Ensure all datasets are in data_config.yaml
    all_dataset_names = list(valid_datasets.keys())
    ensure_datasets_in_config(all_dataset_names)
    
    # Build datasets configuration from auto-discovered datasets
    config['datasets'] = build_datasets_config(valid_datasets, config)
    
    # Update dataset_name to include all datasets
    config['dataset_name'] = '+'.join(all_dataset_names)
    
    print(f"Training on datasets: {', '.join(all_dataset_names)}\n")
    
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=(config["num_workers"] > 0),
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            use_pretrained=config.get("use_pretrained", True),  # Default to True for pretrained weights
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib": 
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    
    # Load model weights
    if "load_run" in config:
        # Option 1: Load from previous training run (fine-tuning your own model)
        load_project_folder = os.path.join("/workspace/model", config["load_run"])
        print(f"🔄 Loading model from {load_project_folder} for fine-tuning...")
        latest_path = os.path.join(load_project_folder, "latest.pth")
        if not os.path.exists(latest_path):
            print(f"❌ Error: Checkpoint not found at {latest_path}")
            print(f"   Available models in /workspace/model/:")
            if os.path.exists("/workspace/model"):
                for item in os.listdir("/workspace/model"):
                    print(f"     - {item}")
            sys.exit(1)
        latest_checkpoint = torch.load(latest_path, weights_only=False)
        load_model(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
        print(f"✅ Loaded checkpoint from epoch {current_epoch - 1}")
    
    elif config.get("use_foundation_model", False):
        # Option 2: Load official foundation model (recommended for first training)
        foundation_path = config.get("foundation_model_path", "/workspace/model")
        model_type = config["model_type"]
        foundation_file = os.path.join(foundation_path, f"{model_type}.pth")
        
        # 嘗試下載 foundation model（如果不存在）
        print(f"\n🔍 Checking for {model_type.upper()} foundation model...")
        if not ensure_foundation_model_exists(model_type, foundation_file):
            print(f"\n❌ ERROR: Failed to download or locate foundation model")
            print(f"   Attempted path: {foundation_file}")
            print(f"\n💡 Available options:")
            print(f"   1. Manually download {model_type}.pth and place it in: {foundation_path}/")
            print(f"   2. Check your internet connection and try again")
            print(f"   3. Set use_foundation_model: false in config to train from scratch")
            raise FileNotFoundError(f"Foundation model {model_type}.pth could not be downloaded or found")
        
        print(f"🎯 Loading official {model_type.upper()} foundation model from {foundation_file}...")
        foundation_checkpoint = torch.load(foundation_file, map_location="cpu", weights_only=False)
        load_model(model, model_type, foundation_checkpoint)
        print(f"✅ Successfully loaded {model_type.upper()} foundation model for fine-tuning")
        print(f"   Starting training from epoch 0 with pretrained weights")
    else:
        print(f"🆕 Training from scratch with ImageNet pretrained EfficientNet backbone")

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            save_visualize=config.get("save_visualize", False),  # Default: False to save disk space
            save_checkpoint_freq=config.get("save_checkpoint_freq", 10),  # Default: every 10 epochs
        )
    else:
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    # Load config from constant variable (can be overridden by command line)
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer - Auto Multi-Dataset Training")
    parser.add_argument(
        "--config",
        "-c",
        default=CONFIG_FILE,
        type=str,
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("\n" + "="*80)
    print("VISUAL NAVIGATION TRANSFORMER TRAINING")
    print("="*80)
    print(f"Config file: {args.config}")
    print(f"Model type: {config.get('model_type', 'unknown')}")
    
    # Get training_name from config
    training_name = config.get('training_name', 'unnamed_training')
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    full_training_name = f"{training_name}_{timestamp}"
    
    # Update run_name to include timestamp
    config["run_name"] = full_training_name
    
    # Create project folder in /workspace/model/ directory (not in train/)
    config["project_folder"] = os.path.join("/workspace/model", full_training_name)
    os.makedirs(config["project_folder"], exist_ok=False)
    
    print(f"Training name: {training_name}")
    print(f"Model output: {config['project_folder']}")

    if config["use_wandb"]:
        wandb.login()
        # Get entity from config or use default
        wandb_entity = config.get('wandb_entity', WANDB_ENTITY)
        # Project name will be based on model type
        model_type = config.get('model_type', 'vint')
        
        wandb_init_kwargs = {
            'project': f"{model_type}-training",
            'settings': wandb.Settings(start_method="fork"),
        }
        
        if wandb_entity is not None:
            wandb_init_kwargs['entity'] = wandb_entity
        
        wandb.init(**wandb_init_kwargs)
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = full_training_name
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print("\nStarting training...")
    print("="*80 + "\n")
    
    main(config)

