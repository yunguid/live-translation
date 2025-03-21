@echo off
echo Installing Real-Time Translation System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check for CUDA
echo Checking CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyTorch and dependencies...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo PyTorch is already installed. Checking CUDA...
    python -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
)

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo To run the application, use:
echo   python real_time_translator.py (for console version)
echo   python gui_translator.py (for graphical version)
echo.
pause 