#!/usr/bin/env python3
"""
Example usage of the TAPIP3D package.

This script demonstrates how to use the package for inference
from another Python project.
"""

import tapip3d
from pathlib import Path

def main():
    # Example 1: Basic usage with video file
    input_video = "demo_inputs/sheep.mp4"
    checkpoint_path = "checkpoints/tapip3d_final.pth"

    if Path(input_video).exists() and Path(checkpoint_path).exists():
        print("Running inference on video file...")
        result_path = tapip3d.run_inference(
            input_path=input_video,
            checkpoint=checkpoint_path,
            output_dir="outputs/example_results",
            device="cuda",
            num_iters=6,
            resolution_factor=2
        )
        print(f"Results saved to: {result_path}")

        # Example: Visualize the results
        print("Opening visualization in browser...")
        tapip3d.visualize(result_path, open_browser=True, block=False)
    else:
        print(f"Input video ({input_video}) or checkpoint ({checkpoint_path}) not found.")
        print("Please ensure you have the required files.")

    # Example 2: Usage with NPZ file (if available)
    input_npz = "demo_inputs/dexycb.npz"

    if Path(input_npz).exists() and Path(checkpoint_path).exists():
        print("\nRunning inference on NPZ file...")
        result_path = tapip3d.run_inference(
            input_path=input_npz,
            checkpoint=checkpoint_path,
            output_dir="outputs/example_results_npz",
            device="cuda",
            num_iters=8,  # Different number of iterations
            support_grid_size=32,  # Larger support grid
        )
        print(f"Results saved to: {result_path}")
    else:
        print(f"\nInput NPZ file ({input_npz}) not found, skipping NPZ example.")

    # Example 3: Minimal usage (no checkpoint provided - will use default if set)
    try:
        print("\nTrying with default checkpoint...")
        result_path = tapip3d.run_inference(
            input_path="path/to/your/video.mp4",  # Replace with actual path
            output_dir="outputs/minimal_example",
        )
        print(f"Results saved to: {result_path}")
    except Exception as e:
        print(f"Expected error (no checkpoint provided): {e}")

    # Example 4: Just visualization (if you already have results)
    print("\n=== Visualization Only Example ===")
    result_file = "outputs/example_results/sheep.result.npz"
    if Path(result_file).exists():
        print(f"Visualizing existing results: {result_file}")
        # Non-blocking visualization
        url = tapip3d.visualize(
            result_file,
            width=512,
            height=384,
            fps=8,
            open_browser=False,  # Don't auto-open browser
            block=False          # Don't block
        )
        print(f"Visualization server started at: {url}")
        print("You can visit this URL in your browser to view the results.")

        # Or process data for custom visualization
        print("\nProcessing data for custom visualization...")
        tapip3d.process_point_cloud_data(
            result_file,
            "custom_viz_data.bin",
            width=320,
            height=240,
            fps=6
        )
        print("Custom visualization data saved to: custom_viz_data.bin")
    else:
        print(f"No existing results found at {result_file}")

if __name__ == "__main__":
    main()