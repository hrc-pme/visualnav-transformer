"""
Model loader and inference logic for GPU container.
Extracted from navigate.ros2.py - no ROS dependencies.
"""
import os
import sys
import subprocess
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Import model utilities (GPU-specific, no ROS dependencies)
sys.path.append('/workspace/deployment/gpu_container')
from utils_gpu import load_model, transform_images, to_numpy
from vint_train.training.train_utils import get_action


# Model download URLs
MODEL_URLS = {
    "gnm": "https://drive.google.com/file/d/1bzCPd_OsXjS2aGPTQladbI8ImxLZwrQh/view?usp=drive_link",
    "vint": "https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view?usp=drive_link", 
    "nomad": "https://drive.google.com/file/d/1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb/view?usp=drive_link"
}


def extract_google_drive_id(url: str) -> Optional[str]:
    """Extract file ID from Google Drive URL"""
    if "drive.google.com" in url:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
    return None


def download_model_from_google_drive(file_id: str, destination: str) -> bool:
    """Download model file from Google Drive"""
    try:
        import gdown
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[GPU] Downloading model to {destination}...")
        gdown.download(download_url, destination, quiet=False)
        return True
    except ImportError:
        print("[GPU] gdown not found. Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            download_url = f"https://drive.google.com/uc?id={file_id}"
            print(f"[GPU] Downloading model to {destination}...")
            gdown.download(download_url, destination, quiet=False)
            return True
        except Exception as e:
            print(f"[GPU] Failed to install gdown or download file: {e}")
            return False
    except Exception as e:
        print(f"[GPU] Failed to download model: {e}")
        return False


def ensure_model_exists(model_name: str, model_path: str) -> bool:
    """Check if model file exists, download if not"""
    if os.path.exists(model_path):
        print(f"[GPU] Model {model_name} already exists at {model_path}")
        return True
    
    print(f"[GPU] Model {model_name} not found at {model_path}")
    
    # Ensure model directory exists
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    if model_name in MODEL_URLS:
        file_id = extract_google_drive_id(MODEL_URLS[model_name])
        if file_id:
            print(f"[GPU] Downloading {model_name} model...")
            if download_model_from_google_drive(file_id, model_path):
                print(f"[GPU] Successfully downloaded {model_name} model")
                return True
            else:
                print(f"[GPU] Failed to download {model_name} model")
                return False
        else:
            print(f"[GPU] Invalid Google Drive URL for {model_name}")
            return False
    else:
        print(f"[GPU] No download URL configured for model: {model_name}")
        return False


class NavigationModel:
    """Handles model loading and inference for navigation"""
    
    def __init__(self, model_name: str, model_config_path: str, device: str = "cuda"):
        """
        Initialize navigation model.
        
        Args:
            model_name: Name of the model (gnm/vint/nomad)
            model_config_path: Path to models.yaml config
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[GPU] Using device: {self.device}")
        
        # Load model configurations
        with open(model_config_path, "r") as f:
            model_paths = yaml.safe_load(f)
        
        if model_name not in model_paths:
            raise ValueError(f"Model '{model_name}' not found in {model_config_path}")
        
        model_config_file = model_paths[model_name]["config_path"]
        # Convert relative path to absolute
        if not os.path.isabs(model_config_file):
            config_dir = os.path.dirname(model_config_path)
            model_config_file = os.path.join(config_dir, model_config_file)
        
        with open(model_config_file, "r") as f:
            self.model_params = yaml.safe_load(f)
        
        # Load model weights
        ckpth_path = model_paths[model_name]["ckpt_path"]
        # Convert relative path to absolute
        if not os.path.isabs(ckpth_path):
            config_dir = os.path.dirname(model_config_path)
            ckpth_path = os.path.join(config_dir, ckpth_path)
        
        if not ensure_model_exists(model_name, ckpth_path):
            raise FileNotFoundError(f"Failed to download or locate model weights for {model_name}")
        
        if not os.path.exists(ckpth_path):
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        
        print(f"[GPU] Loading model from {ckpth_path}")
        self.model = load_model(ckpth_path, self.model_params, self.device).to(self.device).eval()
        
        # Initialize noise scheduler for NOMAD
        if self.model_params["model_type"] == "nomad":
            num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )
        else:
            self.noise_scheduler = None
        
        print(f"[GPU] Model {model_name} loaded successfully")
    
    def get_context_size(self) -> int:
        """Get the context size required by the model"""
        return self.model_params["context_size"]
    
    def get_model_type(self) -> str:
        """Get the model type"""
        return self.model_params["model_type"]
    
    def infer_nomad(
        self,
        context_queue: List[PILImage.Image],
        topomap: List[PILImage.Image],
        start: int,
        end: int,
        close_threshold: int,
        num_samples: int,
        waypoint_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Run inference for NOMAD model.
        
        Returns:
            sampled_actions: Sampled action trajectories
            chosen_waypoint: Selected waypoint
            closest_node: Index of closest node
        """
        if not context_queue or len(context_queue) == 0:
            raise ValueError(f"context_queue is empty in infer_nomad")
        
        obs_images = transform_images(context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(self.device)
        mask = torch.zeros(1).long().to(self.device)
        
        goal_image = [transform_images(img, self.model_params["image_size"], center_crop=False).to(self.device)
                     for img in topomap[start:end+1]]
        goal_image = torch.cat(goal_image, dim=0)
        
        obsgoal_cond = self.model("vision_encoder", obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                                  goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
        dists = to_numpy(self.model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten())
        min_idx = np.argmin(dists)
        closest_node = min_idx + start
        
        sg_idx = min(min_idx + int(dists[min_idx] < close_threshold), len(obsgoal_cond) - 1)
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
        
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(num_samples, 1, 1)
        
        naction = torch.randn((num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
        self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
        
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample
        
        naction = to_numpy(get_action(naction))
        sampled_actions = naction
        chosen_waypoint = naction[0][waypoint_idx]
        
        return sampled_actions, chosen_waypoint, closest_node, dists
    
    def infer_gnm_vint(
        self,
        context_queue: List[PILImage.Image],
        topomap: List[PILImage.Image],
        start: int,
        end: int,
        close_threshold: int,
        waypoint_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Run inference for GNM/ViNT model.
        
        Returns:
            waypoints: Predicted waypoints
            chosen_waypoint: Selected waypoint
            closest_node: Index of closest node
        """
        if not context_queue or len(context_queue) == 0:
            raise ValueError(f"context_queue is empty in infer_gnm_vint")
        
        batch_obs_imgs = [transform_images(context_queue, self.model_params["image_size"]) 
                         for _ in range(end - start + 1)]
        batch_goal_data = [transform_images(topomap[i], self.model_params["image_size"]) 
                          for i in range(start, end + 1)]
        batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(self.device)
        batch_goal_data = torch.cat(batch_goal_data, dim=0).to(self.device)
        
        distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
        distances = to_numpy(distances)
        waypoints = to_numpy(waypoints)
        
        min_dist_idx = np.argmin(distances)
        
        if distances[min_dist_idx] > close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][waypoint_idx]
            closest_node = start + min_dist_idx
        else:
            cur_wp = waypoints[min_dist_idx][waypoint_idx]
            next_idx = min(min_dist_idx + 1, len(waypoints) - 1)
            next_wp = waypoints[next_idx][waypoint_idx]
            
            delta = np.abs(np.array(cur_wp) - np.array(next_wp))
            if np.any(delta < 0.25):
                chosen_waypoint = next_wp
                closest_node = min(start + min_dist_idx + 1, end)
            else:
                chosen_waypoint = cur_wp
                closest_node = start + min_dist_idx
        
        return waypoints, chosen_waypoint, closest_node, distances
    
    def infer(
        self,
        context_queue: List[PILImage.Image],
        topomap: List[PILImage.Image],
        start: int,
        end: int,
        close_threshold: int = 3,
        num_samples: int = 8,
        waypoint_idx: int = 2
    ) -> Dict:
        """
        Run model inference.
        
        Args:
            context_queue: List of context images
            topomap: List of topomap images
            start: Start node index
            end: End node index
            close_threshold: Threshold for node proximity
            num_samples: Number of samples for NOMAD
            waypoint_idx: Waypoint index to use
        
        Returns:
            Dictionary with inference results
        """
        if self.model_params["model_type"] == "nomad":
            sampled_actions, chosen_waypoint, closest_node, distances = self.infer_nomad(
                context_queue, topomap, start, end, close_threshold, num_samples, waypoint_idx
            )
            return {
                'model_type': 'nomad',
                'sampled_actions': sampled_actions,
                'chosen_waypoint': chosen_waypoint,
                'closest_node': closest_node,
                'distances': distances,
                'waypoints': None
            }
        else:
            waypoints, chosen_waypoint, closest_node, distances = self.infer_gnm_vint(
                context_queue, topomap, start, end, close_threshold, waypoint_idx
            )
            return {
                'model_type': self.model_params["model_type"],
                'waypoints': waypoints,
                'chosen_waypoint': chosen_waypoint,
                'closest_node': closest_node,
                'distances': distances,
                'sampled_actions': None
            }
