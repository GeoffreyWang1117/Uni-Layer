"""Setup script for Uni-Layer package"""

from setuptools import setup, find_packages
import os


def read_requirements():
    """Read requirements from requirements.txt"""
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        with open(req_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []


def read_long_description():
    """Read long description from README.md"""
    readme_file = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_file):
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="uni-layer",
    version="0.1.0",
    author="Uni-Layer Team",
    author_email="contact@uni-layer.org",
    description="A Universal Framework for Layer Contribution Analysis Across Deep Learning Architectures",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/GeoffreyWang1117/Uni-Layer",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
        "all": [
            "transformers>=4.20.0",
            "timm>=0.6.0",
            "torch-geometric>=2.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "deep learning",
        "layer analysis",
        "model interpretability",
        "neural networks",
        "model compression",
        "knowledge distillation",
        "pruning",
        "PEFT",
    ],
)
