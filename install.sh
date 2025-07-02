#!/bin/bash

# Installation script for TAPIP3D package

set -e  # Exit on any error

echo "Installing TAPIP3D package..."

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "Warning: Neither conda nor mamba found. Using pip directly."
    CONDA_CMD=""
fi

# Create environment if conda is available
if [ ! -z "$CONDA_CMD" ]; then
    echo "Creating conda environment 'tapip3d'..."
    $CONDA_CMD create -n tapip3d python=3.10 -y
    echo "Activating environment..."
    eval "$($CONDA_CMD shell.bash hook)"
    $CONDA_CMD activate tapip3d

    # Install PyTorch with CUDA support
    echo "Installing PyTorch..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "xformers>=0.0.27" --index-url https://download.pytorch.org/whl/cu124
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
fi

# Install the package in development mode
echo "Installing TAPIP3D package..."
pip install -e .

# Compile pointops2
echo "Compiling pointops2..."
cd tapip3d/third_party/pointops2
if [ ! -z "$CONDA_PREFIX" ]; then
    LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
else
    python setup.py install
fi
cd ../../..

# Compile megasam
echo "Compiling megasam..."
cd tapip3d/third_party/megasam/base
if [ ! -z "$CONDA_PREFIX" ]; then
    LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
else
    python setup.py install
fi
cd ../../../..

echo ""
echo "Installation completed!"
echo ""
echo "To use the package:"
echo "1. If you created a conda environment, activate it:"
echo "   conda activate tapip3d"
echo ""
echo "2. Download checkpoints (see README.md for details)"
echo ""
echo "3. Test the installation:"
echo "   python example_usage.py"
echo ""
echo "4. Or use the command line tools:"
echo "   tapip3d-inference --help"
echo "   tapip3d-visualize --help"