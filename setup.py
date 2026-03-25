"""Setup script for Uni-Layer package (minimal — metadata in pyproject.toml)."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    include_package_data=True,
    zip_safe=False,
)
