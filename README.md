<div align="center">

# TAPIP3D: Tracking Any Point in Persistent 3D Geometry
<a href="https://arxiv.org/abs/2504.14717"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://tapip3d.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>

[Bowei Zhang](https://scholar.google.com/citations?user=tYH72AYAAAAJ)<sup>1,2</sup>*, [Lei Ke](https://www.kelei.site/)<sup>1</sup>\*, [Adam W. Harley](https://adamharley.com/)<sup>3</sup>, [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)<sup>1</sup>

<sup>1</sup>Carnegie Mellon University   &nbsp;  <sup>2</sup>Peking University &nbsp;  <sup>3</sup>Stanford University

\* Equal Contribution

<!-- <a href='https://huggingface.co/spaces/your-username/project'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a> -->

</div>

<img src="./media/teaser1.gif" width="100%" alt="TAPIP3D overview">

## Overview
**TAPIP3D** is a method for long-term **feed-forward** 3D point tracking in monocular RGB and RGB-D video sequences. It introduces a 3D feature cloud representation that lifts image features into a persistent world coordinate space, canceling out camera motion and enabling accurate trajectory estimation across frames.

## Installation

### Install from source

You can install the package directly from source:

```bash
git clone https://github.com/tapip3d/tapip3d.git
cd tapip3d
pip install -e .
```

### Install in development mode

For development, install with optional dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### As a Package

After installation, you can use TAPIP3D in your Python code:

```python
import tapip3d

# Run inference on a video file
result_path = tapip3d.run_inference(
    input_path="path/to/your/video.mp4",
    checkpoint="path/to/checkpoint.pth",
    output_dir="outputs/my_results",
    device="cuda",
    num_iters=6,
    resolution_factor=2
)

print(f"Results saved to: {result_path}")

# Visualize the results
tapip3d.visualize(result_path, open_browser=True)
```

### Command Line Interface

The package also provides command-line tools:

```bash
# Run inference
tapip3d-inference path/to/video.mp4 --checkpoint path/to/checkpoint.pth --output_dir outputs

# Visualize results
tapip3d-visualize path/to/results.npz --port 8080
```

### Function Parameters

#### `run_inference` Function

The `run_inference` function accepts the following parameters:

- `input_path` (str): Path to input video (.mp4, .avi, .mov, .webm) or npz file
- `output_dir` (str, optional): Directory to save results (default: "outputs/inference")
- `checkpoint` (str, optional): Path to model checkpoint
- `device` (str, optional): Device to run inference on (default: "cuda")
- `num_iters` (int, optional): Number of iterations for inference (default: 6)
- `support_grid_size` (int, optional): Grid size for support points (default: 16)
- `num_threads` (int, optional): Number of threads for parallel processing (default: 8)
- `resolution_factor` (int, optional): Resolution scaling factor (default: 2)
- `vis_threshold` (float, optional): Visibility threshold (default: 0.9)
- `depth_model` (str, optional): Depth model to use if depths are not provided (default: "moge")

#### `visualize` Function

The `visualize` function accepts the following parameters:

- `npz_file` (str or Path): Path to the input .result.npz file
- `width` (int, optional): Target width for visualization (default: 256)
- `height` (int, optional): Target height for visualization (default: 192)
- `fps` (int, optional): Base frame rate for playback (default: 4)
- `port` (int, optional): Port to serve on (default: random available port)
- `open_browser` (bool, optional): Whether to automatically open browser (default: True)
- `block` (bool, optional): Whether to block until server is stopped (default: True)

### Supported Input Formats

- Video files: `.mp4`, `.avi`, `.mov`, `.webm`
- NPZ files with pre-computed depths and camera parameters

### Output

The function returns a `Path` object pointing to the saved results NPZ file containing:
- `video`: Original video frames
- `depths`: Depth maps
- `intrinsics`: Camera intrinsic parameters
- `extrinsics`: Camera extrinsic parameters
- `coords`: Tracked 3D coordinates
- `visibs`: Visibility information
- `query_points`: Query points used for tracking

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tapip3d,
  title={TAPIP3D: 3D Point Tracking and Inference},
  author={TAPIP3D Team},
  url={https://tapip3d.github.io/},
  year={2024}
}
```
