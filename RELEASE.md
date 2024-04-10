# Package Release Guide

This document outlines the steps for releasing packages within the MoE-Infinity repository.

## Prerequisites

- **Docker**: Ensure Docker is installed on your system.
- **Repository**: Clone the MoE-Infinity repository to get the latest changes.

    ```bash
    git clone https://github.com/TorchMoE/MoE-Infinity.git
    ```

- **Twine**: Install `twine` for uploading packages to PyPI.

    ```bash
    pip install -U twine
    ```

## Release Steps

1. **Docker Image Pull**: Start by pulling the required NVIDIA CUDA image from Docker Hub.

    ```bash
    docker pull nvidia/cuda:12.1.0-devel-ubuntu20.04
    ```

2. **Build Package**: Execute the command below to build the wheel and source files in the `dist/` directory.

    ```bash
    docker run --gpus all --rm -v $(pwd):/root/workspace -w /root/workspace nvidia/cuda:12.1.0-devel-ubuntu20.04 /bin/bash -c "/root/workspace/matrix_build.sh"
    ```

    > **Note**: Building the wheel file might take some time due to the necessity to build across 4 different Python versions.

3. **Rename Wheel Files**: Adjust the wheel file name to the correct platform tag.

    ```bash
    cd dist
    rename 's/linux/manylinux1/' *.whl
    ```

4. **Upload Package**: Finally, upload the package to either TestPyPI or PyPI as needed.

    - For **TestPyPI**:

        ```bash
        twine upload --repository testpypi dist/*
        ```

    - For **PyPI**:

        ```bash
        twine upload dist/*
        ```

Ensure to follow these steps carefully to successfully release your package. For any issues encountered during the process, refer back to the respective tool's documentation for troubleshooting tips.
