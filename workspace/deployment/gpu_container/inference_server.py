"""
GPU Inference Server
Runs model inference and communicates with ROS2 container via Unix domain socket.
No ROS dependencies.
"""
import os
import sys
import socket
import argparse
from typing import Optional
import numpy as np
from PIL import Image as PILImage

# Add paths
sys.path.append('/workspace/deployment/gpu_container')
sys.path.append('/workspace/deployment/shared')
sys.path.append('/workspace/deployment/src')

from model_loader import NavigationModel
from socket_utils import send_data, recv_data, create_response


SOCKET_PATH = "/tmp/ipc_socket/gpu.sock"
MODEL_CONFIG_PATH = "/workspace/deployment/config/models.yaml"
TOPOMAP_IMAGES_DIR = "/workspace/topomaps"


class InferenceServer:
    """GPU inference server that handles model inference requests"""
    
    def __init__(self, model_name: str, topomap_name: str):
        """
        Initialize inference server.
        
        Args:
            model_name: Name of the model (gnm/vint/nomad)
            topomap_name: Name of the topomap directory
        """
        print(f"[GPU] Initializing inference server with model: {model_name}")
        
        # Load model
        self.model = NavigationModel(model_name, MODEL_CONFIG_PATH)
        self.model_name = model_name
        
        # Load topomap
        topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, topomap_name)
        if not os.path.exists(topomap_dir):
            raise FileNotFoundError(f"Topomap directory not found: {topomap_dir}")
        
        topomap_filenames = sorted(
            os.listdir(topomap_dir),
            key=lambda x: int(x.split(".")[0]),
        )
        self.topomap = [PILImage.open(os.path.join(topomap_dir, fname)) 
                       for fname in topomap_filenames]
        self.num_nodes = len(self.topomap)
        
        print(f"[GPU] Loaded topomap '{topomap_name}' with {self.num_nodes} nodes")
        
        # Socket setup
        self.socket_path = SOCKET_PATH
        self.server_socket: Optional[socket.socket] = None
        
    def start(self):
        """Start the inference server and listen for connections"""
        # Remove existing socket file if it exists
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        # Ensure socket directory exists
        socket_dir = os.path.dirname(self.socket_path)
        os.makedirs(socket_dir, exist_ok=True)
        
        # Create Unix domain socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(1)
        
        print(f"[GPU] Server listening on {self.socket_path}")
        
        try:
            while True:
                print("[GPU] Waiting for connection...")
                client_socket, _ = self.server_socket.accept()
                print("[GPU] Client connected")
                
                try:
                    self.handle_client(client_socket)
                except Exception as e:
                    print(f"[GPU] Error handling client: {e}")
                finally:
                    client_socket.close()
                    print("[GPU] Client disconnected")
        finally:
            self.cleanup()
    
    def handle_client(self, client_socket: socket.socket):
        """Handle a connected client"""
        while True:
            # Receive request
            request = recv_data(client_socket)
            if request is None:
                print("[GPU] Connection closed by client")
                break
            
            # Process request
            response = self.process_request(request)
            
            # Send response
            if not send_data(client_socket, response):
                print("[GPU] Failed to send response")
                break
    
    def process_request(self, request: dict) -> dict:
        """
        Process an inference request.
        
        Args:
            request: Request dictionary with inference parameters
        
        Returns:
            Response dictionary with inference results
        """
        try:
            request_type = request.get('type')
            
            if request_type == 'inference':
                return self.handle_inference(request)
            elif request_type == 'ping':
                return {
                    'type': 'pong', 
                    'success': True,
                    'num_nodes': len(self.topomap) if self.topomap else 0,
                    'context_size': self.model.get_context_size()
                }
            else:
                return create_response(False, error=f"Unknown request type: {request_type}")
        
        except Exception as e:
            print(f"[GPU] Error processing request: {e}")
            return create_response(False, error=str(e))
    
    def handle_inference(self, request: dict) -> dict:
        """
        Handle an inference request.
        
        Args:
            request: Request dictionary containing:
                - context_queue: List of PIL images
                - start_idx: Start node index
                - end_idx: End node index
                - close_threshold: Threshold for node proximity
                - num_samples: Number of samples (for NOMAD)
                - waypoint_idx: Waypoint index to use
        
        Returns:
            Response dictionary with inference results
        """
        context_queue = request.get('context_queue', [])
        start_idx = request.get('start_idx', 0)
        end_idx = request.get('end_idx', 0)
        close_threshold = request.get('close_threshold', 3)
        num_samples = request.get('num_samples', 8)
        waypoint_idx = request.get('waypoint_idx', 2)
        
        # Validate inputs
        if not context_queue or len(context_queue) == 0:
            error_msg = "context_queue is empty"
            print(f"[GPU] Validation error: {error_msg}")
            return create_response(False, error=error_msg)
        
        if start_idx < 0 or end_idx >= self.num_nodes:
            error_msg = f"Invalid node indices: start={start_idx}, end={end_idx}, num_nodes={self.num_nodes}"
            print(f"[GPU] Validation error: {error_msg}")
            return create_response(False, error=error_msg)
        
        if start_idx > end_idx:
            error_msg = f"start_idx ({start_idx}) > end_idx ({end_idx})"
            print(f"[GPU] Validation error: {error_msg}")
            return create_response(False, error=error_msg)
        
        print(f"[GPU] Processing inference: context_queue={len(context_queue)} images, nodes=[{start_idx}:{end_idx}]")
        
        # Run inference
        result = self.model.infer(
            context_queue=context_queue,
            topomap=self.topomap,
            start=start_idx,
            end=end_idx,
            close_threshold=close_threshold,
            num_samples=num_samples,
            waypoint_idx=waypoint_idx
        )
        
        # Create response
        response = create_response(
            success=True,
            distances=result.get('distances'),
            waypoints=result.get('waypoints'),
            sampled_actions=result.get('sampled_actions'),
            closest_node=result.get('closest_node')
        )
        response['model_type'] = result['model_type']
        response['chosen_waypoint'] = result['chosen_waypoint']
        
        # Log inference
        print(f"[GPU] Inference: closest_node={result['closest_node']}, "
              f"waypoint={result['chosen_waypoint']}")
        
        return response
    
    def cleanup(self):
        """Cleanup server resources"""
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        print("[GPU] Server cleaned up")


def main():
    parser = argparse.ArgumentParser(description="GPU Inference Server")
    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        choices=["gnm", "vint", "nomad"],
        help="Model name to use (gnm/vint/nomad)"
    )
    parser.add_argument(
        "--topomap",
        "-t",
        default="6e-dr",
        type=str,
        help="Topomap directory name"
    )
    args = parser.parse_args()
    
    # Create and start server
    server = InferenceServer(args.model, args.topomap)
    server.start()


if __name__ == "__main__":
    main()
