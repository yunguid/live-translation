#!/bin/bash
echo "Installing Real-Time Translation System..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7+ from https://www.python.org/downloads/"
    exit 1
fi

# Check for CUDA
echo "Checking CUDA availability..."
if python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "PyTorch is already installed. Checking CUDA..."
    python3 -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
else
    echo "Installing PyTorch and dependencies..."
    pip3 install torch torchvision torchaudio
fi

echo
echo "Installing dependencies..."
pip3 install -r requirements.txt

echo
echo "Installation complete!"
echo
echo "To run the application, use:"
echo "  python3 real_time_translator.py (for console version)"
echo "  python3 gui_translator.py (for graphical version)"
echo

# Make the scripts executable
chmod +x real_time_translator.py
chmod +x gui_translator.py

echo "Scripts are now executable." 