#!/bin/bash
# Usage: ./install.sh  # klaam
# Usage: ./install.sh --dev # klaam_dev


# arguments handling
dev=false
while test $# -gt 0; do
    case "$1" in
        -h|--help)
            echo "options:"
            echo "-d, --dev=true/false      specify whether it is a dev environment or not"
            exit 0
            ;;
        --dev*)
            dev=true
            shift
            ;;
        -d)
            shift
            dev=true
            shift
            ;;
        *)
            echo "$1 is not a recognized flag!"
            return 1;
            ;;
    esac
done

env_name="klaam"
file_name="environment.yml"
if [ "$dev" == true ]; then
    env_name="klaam_dev"
    file_name="dev_environment.yml"
fi

# reconfigure conda
echo "Setting conda init..."
CONDA_PATH=$(conda info | grep -i 'base environment' | cut -d ":" -f2 | cut -d " " -f2)
source $CONDA_PATH/etc/profile.d/conda.sh

default_install() {
    env_name=$1
    file_name=$2

    echo "Setting environment... (envs/$file_name)"
    conda env create -f .envs/$file_name

    echo "Activating environment... ($env_name)"
    conda activate $env_name

    echo "Upgrading pip..."
    # upgrading pip - https://stackoverflow.com/questions/61365790/error-could-not-build-wheels-for-scipy-which-use-pep-517-and-cannot-be-installe
    python -m pip install pip --upgrade

    echo "Updating poetry config..."
    # disabling poetry's experimental new installer - https://github.com/python-poetry/poetry/issues/4210#issuecomment-877778420
    python -m poetry config experimental.new-installer false

    echo "Installing dependencies using poetry..."
    python -m poetry install

}

default_install $env_name $file_name
