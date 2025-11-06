# Two-Container Architecture

## Overview

This project uses a **two-container architecture** that separates ROS2 logic from GPU compute for better modularity and resource isolation.

## Architecture

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   ROS2 Bridge Container │         │  GPU Inference Container│
│                         │         │                         │
│  - ROS2 Topics          │◄───────►│  - Model Loading        │
│  - Camera Input         │  Unix   │  - PyTorch Inference    │
│  - Velocity Output      │  Socket │  - No ROS Dependencies  │
│  - PD Control           │         │  - CUDA/GPU Access      │
│  - Minimal Dependencies │         │  - Topomap Management   │
└─────────────────────────┘         └─────────────────────────┘
            │                                    │
            └──────────── Shared ────────────────┘
                    /tmp/ipc_socket/
```

### Key Benefits

- **Separation of Concerns**: ROS2 handles topics, GPU handles inference
- **Low Latency**: Unix domain sockets + pickle serialization
- **Resource Isolation**: GPU container gets exclusive CUDA access
- **Clean Dependencies**: No PyTorch in ROS2, no ROS in GPU

## Directory Structure

```
workspace/
├── deployment/
│   ├── ros2_container/
│   │   └── topic_bridge.py          # ROS2 topics + PD control
│   ├── gpu_container/
│   │   ├── inference_server.py      # Socket server
│   │   └── model_loader.py          # Model inference logic
│   ├── shared/
│   │   └── socket_utils.py          # IPC utilities
│   ├── src/
│   │   ├── utils.py                 # Shared utilities
│   │   └── topic_names.py           # Topic definitions
│   └── config/
│       ├── robot.yaml               # Robot config
│       └── models.yaml              # Model paths
└── topomaps/
    ├── 6e-dr/                       # Example topomap
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   └── ...
    └── 6e-elevator/                 # Another topomap
        └── ...
```

## Communication Protocol

### IPC Method
- **Transport**: Unix domain socket (`/tmp/ipc_socket/gpu.sock`)
- **Serialization**: Pickle (Python native)
- **Message Format**: Length-prefixed (4-byte header + payload)

### Request (ROS2 → GPU)
```python
{
    'type': 'inference',
    'context_queue': [PIL.Image, ...],  # Observation images
    'start_idx': int,                    # Start node
    'end_idx': int,                      # End node
    'close_threshold': int,              # Node proximity threshold
    'num_samples': int,                  # NOMAD samples
    'waypoint_idx': int                  # Waypoint index
}
```

### Response (GPU → ROS2)
```python
{
    'success': bool,
    'closest_node': int,
    'chosen_waypoint': np.ndarray,       # Selected waypoint
    'distances': np.ndarray,             # Distance predictions
    'waypoints': np.ndarray,             # GNM/ViNT waypoints
    'sampled_actions': np.ndarray,       # NOMAD sampled actions
    'model_type': str,                   # 'gnm'/'vint'/'nomad'
    'error': str                         # Error message if failed
}
```

## How to Run

### GPU Container

Start the inference server:

```bash
cd /workspace/deployment/gpu_container
python3 inference_server.py --model vint --topomap 6e-dr
```

**Arguments:**
- `--model`, `-m`: Model name (`gnm`/`vint`/`nomad`) [default: vint]
- `--topomap`, `-t`: Topomap directory name [default: 6e-dr]

### ROS2 Container

Start the topic bridge:

```bash
source /opt/ros/humble/setup.bash
cd /workspace/deployment/ros2_container
python3 topic_bridge.py --waypoint 2 --start-node 0 --goal-node -1
```

**Arguments:**
- `--waypoint`, `-w`: Waypoint index [default: 2]
- `--start-node`, `-s`: Start node index [default: 0]
- `--goal-node`, `-g`: Goal node (-1 for last) [default: -1]
- `--radius`, `-r`: Local search radius [default: 4]
- `--close-threshold`, `-t`: Proximity threshold [default: 3]
- `--num-samples`, `-n`: NOMAD samples [default: 8]

**Note**: Start GPU container first, then ROS2 container.

## ROS2 Topics

### Published
- `/waypoint` (Float32MultiArray): Selected waypoint
- `/vn/candidate_waypoints` (Float32MultiArray): Candidate waypoints
- `/vn/chosen_waypoint` (Float32MultiArray): Chosen waypoint
- `/sampled_actions` (Float32MultiArray): Sampled actions (NOMAD)
- `/vn/node` (Int32): Current node index
- `/vn/start_node` (Int32): Start node
- `/vn/end_node` (Int32): End node
- `/reach_goal` (Bool): Goal reached status
- `/goal_pose` (PoseStamped): Target pose
- `${VEL_TOPIC}` (Twist): Velocity commands
- `camera/image/visualnav` (Image): Processed camera

### Subscribed
- `/camera/camera/color/image_raw` (Image): Camera input
