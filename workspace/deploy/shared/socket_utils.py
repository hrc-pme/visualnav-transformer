"""
Shared socket utilities for IPC between ROS2 and GPU containers.
Uses pickle for serialization and Unix domain sockets for low-latency communication.
"""
import pickle
import struct
import socket
from typing import Any, Dict, Optional
import numpy as np


def serialize_data(data: Any) -> bytes:
    """
    Serialize data using pickle.
    
    Args:
        data: Any Python object (typically dict with numpy arrays)
    
    Returns:
        Serialized bytes with length prefix
    """
    serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    # Prefix with 4-byte length for safe receiving
    return struct.pack('>I', len(serialized)) + serialized


def deserialize_data(data: bytes) -> Any:
    """
    Deserialize data from pickle bytes.
    
    Args:
        data: Pickled bytes
    
    Returns:
        Deserialized Python object
    """
    return pickle.loads(data)


def send_data(sock: socket.socket, data: Any) -> bool:
    """
    Send serialized data through socket with length prefix.
    
    Args:
        sock: Connected socket
        data: Data to send
    
    Returns:
        True if successful, False otherwise
    """
    try:
        serialized = serialize_data(data)
        sock.sendall(serialized)
        return True
    except Exception as e:
        print(f"[SOCKET] Failed to send data: {e}")
        return False


def recv_exact(sock: socket.socket, size: int) -> Optional[bytes]:
    """
    Receive exact number of bytes from socket.
    
    Args:
        sock: Connected socket
        size: Number of bytes to receive
    
    Returns:
        Received bytes or None if connection closed
    """
    data = b''
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def recv_data(sock: socket.socket) -> Optional[Any]:
    """
    Receive serialized data from socket with length prefix.
    
    Args:
        sock: Connected socket
    
    Returns:
        Deserialized data or None if error/connection closed
    """
    try:
        # Read 4-byte length prefix
        length_data = recv_exact(sock, 4)
        if not length_data:
            return None
        
        length = struct.unpack('>I', length_data)[0]
        
        # Read the actual data
        serialized = recv_exact(sock, length)
        if not serialized:
            return None
        
        return deserialize_data(serialized)
    except Exception as e:
        print(f"[SOCKET] Failed to receive data: {e}")
        return None


def create_request(
    obs_images: np.ndarray,
    goal_images: np.ndarray,
    context_queue: list,
    start_idx: int,
    end_idx: int,
    model_type: str,
    **kwargs
) -> Dict:
    """
    Create a standardized inference request.
    
    Args:
        obs_images: Observation images as numpy array
        goal_images: Goal images as numpy array
        context_queue: List of PIL images for context
        start_idx: Start node index
        end_idx: End node index
        model_type: Type of model (gnm/vint/nomad)
        **kwargs: Additional model-specific parameters
    
    Returns:
        Dictionary containing request data
    """
    return {
        'type': 'inference',
        'obs_images': obs_images,
        'goal_images': goal_images,
        'context_queue': context_queue,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'model_type': model_type,
        **kwargs
    }


def create_response(
    success: bool,
    distances: Optional[np.ndarray] = None,
    waypoints: Optional[np.ndarray] = None,
    sampled_actions: Optional[np.ndarray] = None,
    closest_node: Optional[int] = None,
    error: Optional[str] = None
) -> Dict:
    """
    Create a standardized inference response.
    
    Args:
        success: Whether inference succeeded
        distances: Distance predictions
        waypoints: Waypoint predictions
        sampled_actions: Sampled actions (for NOMAD)
        closest_node: Closest node index
        error: Error message if failed
    
    Returns:
        Dictionary containing response data
    """
    response = {
        'success': success,
        'error': error
    }
    
    if success:
        response.update({
            'distances': distances,
            'waypoints': waypoints,
            'sampled_actions': sampled_actions,
            'closest_node': closest_node
        })
    
    return response
