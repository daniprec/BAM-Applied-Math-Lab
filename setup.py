from setuptools import find_packages, setup

setup(
    name="amlab",
    version="0.1.0",
    description="Applied Math Lab: models and utilities for collective motion, ODEs, PDEs, and more.",
    author="Daniel Precioso",
    author_email="daniel.precioso@ie.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        # Add other dependencies as needed
    ],
    python_requires=">=3.11",
    include_package_data=True,
    url="https://github.com/daniprec/BAM-Applied-Math-Lab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
