# TAPIP3D Package Setup Summary

This document summarizes the changes made to convert TAPIP3D into an installable package.

## Changes Made

### 1. Package Structure
- Moved all source code into `tapip3d/` directory
- Created proper Python package structure with `__init__.py` files
- Organized modules: `annotation/`, `datasets/`, `models/`, `utils/`, `third_party/`, `training/`

### 2. Refactored Inference Function
- **Before**: Logic was embedded in `if __name__ == "__main__"` block
- **After**: Extracted into reusable `run_inference()` function
- **Function Signature**:
  ```python
  def run_inference(
      input_path: str,
      output_dir: str = "outputs/inference",
      checkpoint: Optional[str] = None,
      device: str = "cuda",
      num_iters: int = 6,
      support_grid_size: int = 16,
      num_threads: int = 8,
      resolution_factor: int = 2,
      vis_threshold: Optional[float] = 0.9,
      depth_model: str = "moge"
  ) -> Path
  ```

### 3. Fixed Import System
- Converted all absolute imports to relative imports within the package
- Fixed circular import issues
- Created import fixing script that updated 37+ files automatically
- All internal imports now use relative paths (e.g., `from .utils import ...`)

### 4. Created Installation Files
- **`setup.py`**: Traditional setuptools configuration
- **`pyproject.toml`**: Modern Python packaging configuration
- **`install.sh`**: Automated installation script for dependencies
- Both configurations support:
  - Package installation with `pip install -e .`
  - Command-line tools: `tapip3d-inference` and `tapip3d-visualize`
  - Development dependencies with `pip install -e ".[dev]"`

### 5. Updated Documentation
- **`README.md`**: Added installation and usage instructions
- **`example_usage.py`**: Demonstrates how to use the package from Python code
- **`PACKAGE_SETUP.md`**: This summary document

### 6. Command Line Interface
- Preserved original CLI functionality
- Added entry points for easy access:
  - `tapip3d-inference` → runs `tapip3d.inference:main`
  - `tapip3d-visualize` → runs `tapip3d.visualize:main`

## Usage Examples

### As a Python Package
```python
import tapip3d

# Run inference
result_path = tapip3d.run_inference(
    input_path="video.mp4",
    checkpoint="model.pth",
    output_dir="my_results"
)
print(f"Results saved to: {result_path}")
```

### Command Line
```bash
# Install the package
pip install -e .

# Run inference
tapip3d-inference video.mp4 --checkpoint model.pth --output_dir results

# Visualize results
tapip3d-visualize results/video.result.npz
```

### From Another Project
```python
# In requirements.txt
# git+https://github.com/tapip3d/tapip3d.git

# In your code
import tapip3d
result = tapip3d.run_inference("path/to/video.mp4", checkpoint="model.pth")
```

## Key Benefits

1. **Reusability**: Can be imported and used from other Python projects
2. **Clean API**: Single function call handles all inference logic
3. **Flexibility**: All parameters are configurable with sensible defaults
4. **Distribution**: Can be installed via pip, uploaded to PyPI, etc.
5. **Maintainability**: Proper package structure makes code organization clearer
6. **Backward Compatibility**: Original CLI interface still works

## Files Created/Modified

### New Files
- `setup.py` - Package configuration
- `pyproject.toml` - Modern package configuration
- `install.sh` - Installation script
- `example_usage.py` - Usage examples
- `PACKAGE_SETUP.md` - This documentation

### Modified Files
- `tapip3d/__init__.py` - Package entry point, exports `run_inference`
- `tapip3d/inference.py` - Refactored to extract reusable function
- `tapip3d/visualize.py` - Updated imports, fixed errno issue
- `README.md` - Added installation and usage instructions
- **37+ Python files** - Fixed imports to use relative imports

### Moved Files
- `inference.py` → `tapip3d/inference.py`
- `visualize.py` → `tapip3d/visualize.py`
- All source directories moved into `tapip3d/`

## Installation

```bash
# Clone and install
git clone <repository>
cd tapip3d
pip install -e .

# Or use the installation script
./install.sh
```

The package is now ready for distribution and can be used both as a library and command-line tool.