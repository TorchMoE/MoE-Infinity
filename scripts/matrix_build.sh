# get conda from miniconda
apt update && apt install wget -y

cd ~ && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda

# define a list of strings
PYTHON_VERSIONS="38 39 310 311"
declare -a PYTHON_VERSIONS_FOR_CONDA
PYTHON_VERSIONS_FOR_CONDA[38]="3.8"
PYTHON_VERSIONS_FOR_CONDA[39]="3.9"
PYTHON_VERSIONS_FOR_CONDA[310]="3.10"
PYTHON_VERSIONS_FOR_CONDA[311]="3.11"

cd /root/workspace

for version in $PYTHON_VERSIONS
do
    echo "Creating environment for Python $version ${PYTHON_VERSIONS_FOR_CONDA[$version]}"
    /miniconda/bin/conda create -n py$version python=${PYTHON_VERSIONS_FOR_CONDA[$version]} -y
    /miniconda/envs/py$version/bin/pip install --upgrade pip
    echo "Complete environment creation for Python $version"
done

for version in $PYTHON_VERSIONS
do
    echo "Building for Python $version"
    /miniconda/envs/py$version/bin/pip install -r requirements.txt
    /miniconda/envs/py$version/bin/pip install build

    echo "Complete requirements installation for Python $version"
    BUILD_OPS=1 /miniconda/envs/py$version/bin/python -m build

    echo "Complete build for Python $version"
done