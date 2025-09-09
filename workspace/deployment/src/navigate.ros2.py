import argparse
import os
import sys
import time
import subprocess
import urllib.request
import shutil

import numpy as np
import rclpy
import torch
import yaml
from cv_bridge import CvBridge
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from geometry_msgs.msg import Twist
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from topic_names import IMAGE_TOPIC, SAMPLED_ACTIONS_TOPIC, WAYPOINT_TOPIC
from utils import load_model, msg_to_pil, to_numpy, transform_images
from vint_train.training.train_utils import get_action

# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps"
TOPOMAP_NAME = "6e6"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
Z_RATIO = 0.5  # scale z (yaw) speed to half
XY_RATIO = 0.5

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
VEL_TOPIC = robot_config["vel_navi_topic"]

# GLOBALS
context_queue = []
context_size = None

# Model download URLs
MODEL_URLS = {
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
        print(f"Downloading model to {destination}...")
        gdown.download(download_url, destination, quiet=False)
        return True
    except ImportError:
        print("gdown not found. Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            download_url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading model to {destination}...")
            gdown.download(download_url, destination, quiet=False)
            return True
        except Exception as e:
            print(f"Failed to install gdown or download file: {e}")
            return False
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False

def ensure_model_exists(model_name, model_path):
    """檢查模型檔案是否存在，如果不存在則下載"""
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
        return True
    
    print(f"Model {model_name} not found at {model_path}")
    
    # 確保模型目錄存在
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    if model_name in MODEL_URLS:
        file_id = extract_google_drive_id(MODEL_URLS[model_name])
        if file_id:
            print(f"Downloading {model_name} model...")
            if download_model_from_google_drive(file_id, model_path):
                print(f"Successfully downloaded {model_name} model")
                return True
            else:
                print(f"Failed to download {model_name} model")
                return False
        else:
            print(f"Invalid Google Drive URL for {model_name}")
            return False
    else:
        print(f"No download URL configured for model: {model_name}")
        return False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg):
    obs_img = msg_to_pil(msg).rotate(270, expand=True)

    if 'node' in globals():
        try:
            cv_img = np.array(obs_img)
            image_msg = node.bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
            image_msg.header.stamp = node.get_clock().now().to_msg()
            node.image_pub.publish(image_msg)
        except Exception as e:
            print(f"Failed to publish visualnav image: {e}")

    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


class NavigationNode(Node):
    def __init__(self):
        super().__init__("navigation_node")
        self.create_subscription(Image, "/camera/camera/color/image_raw", callback_obs, qos_profile_sensor_data)
        qos = QoSProfile(depth=10)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, qos)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, qos)
        self.image_pub = self.create_publisher(Image, "camera/image/visualnav", qos_profile_sensor_data)
        self.reach_goal_pub = self.create_publisher(Bool, "/reach_goal", qos)
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, qos)  # 添加速度控制發布器
        self.bridge = CvBridge()
        
    def publish_zero_velocity(self):
        """發布零速度指令以停止機器人"""
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.linear.z = 0.0
        stop_cmd.angular.x = 0.0
        stop_cmd.angular.y = 0.0
        stop_cmd.angular.z = 0.0
        self.vel_pub.publish(stop_cmd)
        self.get_logger().info("Published zero velocity command to stop robot")

def main(args: argparse.Namespace):
    global context_size

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    ckpth_path = model_paths[args.model]["ckpt_path"]
    
    # 檢查並下載模型（如果需要）
    if not ensure_model_exists(args.model, ckpth_path):
        raise FileNotFoundError(f"Failed to download or locate model weights for {args.model} at {ckpth_path}")
    
    if not os.path.exists(ckpth_path):
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    print(f"Loading model from {ckpth_path}")

    model = load_model(ckpth_path, model_params, device).to(device).eval()

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    topomap_filenames = sorted(
        os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, TOPOMAP_NAME)),
        key=lambda x: int(x.split(".")[0]),
    )
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{TOPOMAP_NAME}"
    topomap = [PILImage.open(os.path.join(topomap_dir, fname)) for fname in topomap_filenames]
    num_nodes = len(topomap)
    
    assert -1 <= args.goal_node < num_nodes, "Invalid goal index"
    goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node

    rclpy.init()
    global node
    node = NavigationNode()
    closest_node = 0
    reached_goal = False
    start, end = -1, -1

    while rclpy.ok():
        loop_start_time = time.time()
        chosen_waypoint = np.zeros(4)

        # 如果已經到達目標，持續發布停止信號
        if reached_goal:
            node.publish_zero_velocity()
            node.reach_goal_pub.publish(Bool(data=True))
            print("[Navigation] Goal reached. Robot stopped.")
            time.sleep(max(0, (1.0 / RATE) - (time.time() - loop_start_time)))
            rclpy.spin_once(node, timeout_sec=0)
            continue

        if len(context_queue) > model_params["context_size"]:
            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)

            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(device)
                mask = torch.zeros(1).long().to(device)

                goal_image = [transform_images(img, model_params["image_size"], center_crop=False).to(device)
                            for img in topomap[start:end+1]]
                goal_image = torch.cat(goal_image, dim=0)

                obsgoal_cond = model("vision_encoder", obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                                    goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = to_numpy(model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start

                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                naction = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    noise_pred = model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)
                    naction = noise_scheduler.step(noise_pred, k, naction).prev_sample

                naction = to_numpy(get_action(naction))
                node.sampled_actions_pub.publish(Float32MultiArray(data=np.concatenate(([0], naction.flatten())).tolist()))
                chosen_waypoint = naction[0][args.waypoint]

            else:
                batch_obs_imgs = [transform_images(context_queue, model_params["image_size"]) for _ in range(end - start + 1)]
                batch_goal_data = [transform_images(topomap[i], model_params["image_size"]) for i in range(start, end + 1)]
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)

                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] > args.close_threshold:
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    cur_wp = waypoints[min_dist_idx][args.waypoint]
                    next_idx = min(min_dist_idx + 1, len(waypoints) - 1)
                    next_wp = waypoints[next_idx][args.waypoint]

                    delta = np.abs(np.array(cur_wp) - np.array(next_wp))
                    if np.any(delta < 0.25):
                        chosen_waypoint = next_wp
                        closest_node = min(start + min_dist_idx + 1, goal_node)
                    else:
                        chosen_waypoint = cur_wp
                        closest_node = start + min_dist_idx

        # Normalize and scale XY (linear) and Z (yaw)
        if model_params["normalize"]:
            chosen_waypoint[0:2] *= MAX_V / RATE * XY_RATIO
            chosen_waypoint[2] *= MAX_W / RATE * Z_RATIO

        waypoint_msg = Float32MultiArray(data=chosen_waypoint.tolist())
        node.waypoint_pub.publish(waypoint_msg)
        
        # 檢查是否到達目標
        goal_reached = bool(closest_node == goal_node)
        node.reach_goal_pub.publish(Bool(data=goal_reached))

        waypoint_str = f"[{chosen_waypoint[0]:.2f} {chosen_waypoint[1]:.2f} {chosen_waypoint[2]:.2f}]"
        print(f"[Status] Node: {closest_node}/{goal_node} | Ref Node: {start} to {end} | Waypoint: {waypoint_str}")

        if goal_reached:
            print("[Navigation] Goal reached. Stopping robot...")
            # 持續發送零速度指令停止機器人
            node.publish_zero_velocity()
            # 繼續循環以持續發布停止信號，而不是立即退出
            reached_goal = True

        time.sleep(max(0, (1.0 / RATE) - (time.time() - loop_start_time)))
        rclpy.spin_once(node, timeout_sec=0)

    node.destroy_node()
    rclpy.shutdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        help="model name (only nomad is supported) (hint: check config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
