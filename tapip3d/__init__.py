# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

"""
TAPIP3D: 3D Point Tracking and Inference Package

This package provides functionality for 3D point tracking and inference
on video sequences with depth information.
"""

from .inference import run_inference
from .visualize import visualize, process_point_cloud_data

__version__ = "0.1.0"
__author__ = "TAPIP3D Team"
__email__ = "contact@tapip3d.github.io"

__all__ = [
    "run_inference",
    "visualize",
    "process_point_cloud_data",
]