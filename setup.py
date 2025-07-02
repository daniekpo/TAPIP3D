#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read the README file for long description
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="tapip3d",
    version="0.1.0",
    description="TAPIP3D: 3D Point Tracking and Inference Package",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="TAPIP3D Team",
    url="https://tapip3d.github.io/",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'tapip3d-inference=tapip3d.inference:main',
            'tapip3d-visualize=tapip3d.visualize:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    include_package_data=True,
    package_data={
        'tapip3d': ['utils/viz.html'],
    },
)