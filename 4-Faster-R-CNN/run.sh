#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "Starting Faster R-CNN Object Detector Web Application..."
echo "=========================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH." >&2
    exit 1
fi

# Check if conda environment 'sliding_window_env' exists
if ! conda env list | grep -qE '^sliding_window_env[[:space:]]'; then
    echo "Error: Conda environment 'sliding_window_env' does not exist." >&2
    echo "Please create it using: conda create -n sliding_window_env python=3.9 -y" >&2
    exit 1
fi

PYTHON_CMD=(conda run -n sliding_window_env --no-capture-output python)

# Check for required packages
echo "Verifying Python dependencies..."
"${PYTHON_CMD[@]}" -c "
libs = ['flask', 'torch', 'cv2', 'numpy']
missing = []
for lib in libs:
    try:
        __import__(lib)
    except ImportError:
        missing.append(lib)
if missing:
    print('Missing:', ', '.join(missing))
    exit(1)
" || {
    echo "Error: Some required Python libraries are missing from the conda environment." >&2
    echo "Please install them using: conda run -n sliding_window_env pip install flask torch opencv-python numpy" >&2
    exit 1
}

echo "All dependencies verified successfully."
echo "Launching Flask backend for Faster R-CNN..."
echo "Open your browser and navigate to: http://127.0.0.1:5004"
echo "Press Ctrl+C to stop the server."
echo "----------------------------------------------------------"

# Run the app
"${PYTHON_CMD[@]}" app.py
