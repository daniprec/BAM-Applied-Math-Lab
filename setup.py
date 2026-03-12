from pathlib import Path

from setuptools import find_namespace_packages, setup

README = Path("README_PYPI.md").read_text(encoding="utf-8")

setup(
    name="amlab",
    version="0.1.0",
    description="Applied Math Lab: models and utilities for collective motion, ODEs, PDEs, and more.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Daniel Precioso",
    author_email="daniel.precioso@ie.edu",
    packages=find_namespace_packages(include=["amlab*"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "networkx",
        "pandas",
        "pillow",
    ],
    extras_require={
        "site": [
            "ipykernel",
            "jupyter_client",
            "nbclient",
            "nbformat",
            "pyyaml",
            "plotly",
            "streamlit",
        ]
    },
    python_requires=">=3.11",
    include_package_data=True,
    url="https://github.com/daniprec/BAM-Applied-Math-Lab",
    project_urls={
        "Source": "https://github.com/daniprec/BAM-Applied-Math-Lab",
        "Course Site": "https://daniprec.github.io/BAM-Applied-Math-Lab/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
