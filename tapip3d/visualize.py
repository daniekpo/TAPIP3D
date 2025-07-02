# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

import os
import numpy as np
import cv2
import json
import struct
import zlib
import argparse
from einops import rearrange
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import http.server
import socketserver
import socket
import sys
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer
import errno
import webbrowser
import time
import threading
from typing import Union, Optional

viz_html_path = Path(__file__).parent / "utils" / "viz.html"
DEFAULT_PORT = 8000

def compress_and_write(filename, header, blob):
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)

def process_point_cloud_data(npz_file: Union[str, Path], output_file: Union[str, Path], width: int = 256, height: int = 192, fps: int = 4):
    """
    Process point cloud data from an NPZ file for visualization.

    Args:
        npz_file: Path to the input .result.npz file
        output_file: Path to save the processed data.bin file
        width: Target width for visualization (default: 256)
        height: Target height for visualization (default: 192)
        fps: Base frame rate for playback (default: 4)
    """
    fixed_size = (width, height)

    data = np.load(npz_file)
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    trajs = data["coords"]
    T, C, H, W = data["video"].shape

    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180 / np.pi)
    original_aspect_ratio = (W / fx) / (H / fy)

    rgb_video = (rearrange(data["video"], "T C H W -> T H W C") * 255).astype(np.uint8)
    rgb_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)
                          for frame in rgb_video])

    depth_video = data["depths"].astype(np.float32)
    depth_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_NEAREST)
                            for frame in depth_video])

    scale_x = fixed_size[0] / W
    scale_y = fixed_size[1] / H
    intrinsics = intrinsics.copy()
    intrinsics[:, 0, :] *= scale_x
    intrinsics[:, 1, :] *= scale_y

    min_depth = float(depth_video.min()) * 0.8
    max_depth = float(depth_video.max()) * 1.5

    depth_normalized = (depth_video - min_depth) / (max_depth - min_depth)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)

    depths_rgb = np.zeros((T, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)

    first_frame_inv = np.linalg.inv(extrinsics[0])
    normalized_extrinsics = np.array([first_frame_inv @ ext for ext in extrinsics])

    normalized_trajs = np.zeros_like(trajs)
    for t in range(T):
        homogeneous_trajs = np.concatenate([trajs[t], np.ones((trajs.shape[1], 1))], axis=1)
        transformed_trajs = (first_frame_inv @ homogeneous_trajs.T).T
        normalized_trajs[t] = transformed_trajs[:, :3]

    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics,
        "extrinsics": normalized_extrinsics,
        "inv_extrinsics": np.linalg.inv(normalized_extrinsics),
        "trajectories": normalized_trajs.astype(np.float32),
        "cameraZ": 0.0
    }

    header = {}
    blob_parts = []
    offset = 0
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "offset": offset,
            "length": len(arr_bytes)
        }
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)

    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=9)

    header["meta"] = {
        "depthRange": [min_depth, max_depth],
        "totalFrames": int(T),
        "resolution": fixed_size,
        "baseFrameRate": fps,
        "numTrajectoryPoints": normalized_trajs.shape[1],
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "original_aspect_ratio": float(original_aspect_ratio),
        "fixed_aspect_ratio": float(fixed_size[0]/fixed_size[1])
    }

    compress_and_write(output_file, header, compressed_blob)

def visualize(
    npz_file: Union[str, Path],
    width: int = 256,
    height: int = 192,
    fps: int = 4,
    port: Optional[int] = None,
    open_browser: bool = True,
    block: bool = True
) -> str:
    """
    Visualize TAPIP3D results in a web browser.

    Args:
        npz_file: Path to the input .result.npz file
        width: Target width for visualization (default: 256)
        height: Target height for visualization (default: 192)
        fps: Base frame rate for playback (default: 4)
        port: Port to serve on (default: random available port)
        open_browser: Whether to automatically open browser (default: True)
        block: Whether to block until server is stopped (default: True)

    Returns:
        URL of the visualization server

    Example:
        >>> import tapip3d
        >>> url = tapip3d.visualize("results.npz", open_browser=True)
        >>> print(f"Visualization available at: {url}")
    """
    if port is None:
        port = DEFAULT_PORT

    temp_dir = TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    # Process data
    process_point_cloud_data(
        npz_file,
        temp_path / "data.bin",
        width=width,
        height=height,
        fps=fps
    )
    shutil.copy(viz_html_path, temp_path / "index.html")

    # Start server
    os.chdir(temp_path)

    host = "127.0.0.1"
    Handler = SimpleHTTPRequestHandler
    httpd = None

    try:
        httpd = ThreadingTCPServer((host, port), Handler)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(f"Port {port} is already in use, trying a random port...")
            try:
                httpd = ThreadingTCPServer((host, 0), Handler)
                port = httpd.server_address[1] # Get the assigned port
            except OSError as e2:
                print(f"Failed to bind to a random port: {e2}", file=sys.stderr)
                raise
        else:
            print(f"Failed to start server: {e}", file=sys.stderr)
            raise

    url = f"http://{host}:{port}"
    print(f"Serving visualization at {url}")

    if open_browser:
        # Open browser in a separate thread
        def open_browser_delayed():
            time.sleep(1)  # Give server time to start
            webbrowser.open(url)

        threading.Thread(target=open_browser_delayed, daemon=True).start()

    if block:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nVisualization server stopped.")
        finally:
            httpd.server_close()
            temp_dir.cleanup()
    else:
        # Return the server and temp_dir so they can be managed externally
        return url, httpd, temp_dir

    return url

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to the input .result.npz file')
    parser.add_argument('--width', '-W', type=int, default=256, help='Target width')
    parser.add_argument('--height', '-H', type=int, default=192, help='Target height')
    parser.add_argument('--fps', type=int, default=4, help='Base frame rate for playback')
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help=f'Port to serve the visualization (default: {DEFAULT_PORT})')

    args = parser.parse_args()

    # Use the programmatic interface
    visualize(
        npz_file=args.input_file,
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.port,
        open_browser=True,
        block=True
    )

if __name__ == "__main__":
    main()