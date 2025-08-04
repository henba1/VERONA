#!/bin/bash

# Script to setup a test virtual environment for ada-verona testing
# Usage: ./setup_test_venv.sh <venv_folder_path> <ada_auto_verify_path>

set -e  # Exit on any error

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <venv_folder_path> <ada_auto_verify_path>"
    echo "Example: $0 /path/to/venv/folder /path/to/ada-auto-verify"
    exit 1
fi

VENV_FOLDER="$1"
ADA_AUTO_VERIFY_PATH="$2"

# Validate paths
if [ ! -d "$VENV_FOLDER" ]; then
    echo "Error: Virtual environment folder '$VENV_FOLDER' does not exist"
    exit 1
fi

if [ ! -d "$ADA_AUTO_VERIFY_PATH" ]; then
    echo "Error: ada-auto-verify path '$ADA_AUTO_VERIFY_PATH' does not exist"
    exit 1
fi

# Function to find next available test number
find_next_test_number() {
    local base_folder="$1"
    local test_num=1
    
    while [ -d "$base_folder/av_test_$test_num" ]; do
        ((test_num++))
    done
    
    echo $test_num
}

# Find the next available test number
TEST_NUM=$(find_next_test_number "$VENV_FOLDER")
VENV_NAME="av_test_$TEST_NUM"
VENV_PATH="$VENV_FOLDER/$VENV_NAME"

echo "Setting up test virtual environment: $VENV_NAME"
echo "Virtual environment path: $VENV_PATH"
echo "ada-auto-verify path: $ADA_AUTO_VERIFY_PATH"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv "$VENV_PATH"

# Load required modules
echo "Loading required modules..."
module load GCC/11.3.0
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
module load Python/3.10.4
module save verona-modules

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip and install build tools
echo "Upgrading pip and installing build tools..."
uv pip install --upgrade pip setuptools wheel build

# Install ada-verona from Test PyPI
echo "Installing ada-verona 0.1.7 from Test PyPI..."
uv pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ada-verona==0.1.7

# Install ada-auto-verify
echo "Installing ada-auto-verify..."
cd "$ADA_AUTO_VERIFY_PATH"
pip install build
python -m build --wheel

echo "Test virtual environment setup complete!"
echo "Virtual environment: $VENV_PATH"
echo "To activate: source $VENV_PATH/bin/activate" 