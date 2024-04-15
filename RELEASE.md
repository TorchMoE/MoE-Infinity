# Package Release Guide

This document describes the process of releasing a new version of the MoE-Infinity-Rel package.

## Automated Release Process

The release mechanism is fully automated through a GitHub Actions workflow, which is defined in the `.github/workflows/publish.yml` file. This workflow triggers upon the creation and publication of a new version tag formatted as `v*` within the repository.

### Steps to Release a New Version
To release a new version, such as version 1.0.0, please adhere to the following procedure:

1. Update Version: Modify the version number in the setup.py file to reflect the new release version.
2. Commit Changes: Commit these changes with an appropriate commit message that summarizes the update, such as "Update version for 1.0.0 release".
3. Create and Push Tag: Tag the latest commit with the new version number and push the tag to the repository. Use the following commands to accomplish this:
    ```bash
    git tag v1.0.0
    git push origin v1.0.0
    ```

Upon the successful push of the tag, the workflow will creata a new release draft, build the package and publish it to the GitHub Package Registry and PyPI repositories.


## Manual Package Building and Publishing

For developers who prefer to manually build and publish their package to PyPI, the following steps provide a detailed guide to execute this process effectively.

1. Start by cloning the repository and navigating to the root directory of the package:
    ```bash
    git clone https://github.com/TorchMoE/MoE-Infinity.git
    cd MoE-Infinity
    ```
2. Install the required dependencies to build the package:
    ```bash
    pip install -r requirements.txt
    pip install build 
    ```
3. Build the source distribution and wheel for the package using:
    ```bash
    BUILD_OPS=1 python -m build
    ```
    This command generates the package files in the `dist/` directory.
4. Upload the built package to the PyPI repository using `twine`:
    ```bash
    twine upload dist/*
    ```
    Ensure that you have the necessary credentials configured for `twine` to authenticate to PyPI.


To build the package wheel for multiple Python versions, you should execute the build process individually for each version by specifying the corresponding Python interpreter. 