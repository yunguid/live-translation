# Real-time Translation App Setup and Run Script for Windows
# ======================================================

Write-Host "======================================================"
Write-Host "  Real-time Translation App - Windows Setup and Run   " 
Write-Host "======================================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✓ Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.8 or newer." -ForegroundColor Red
    Write-Host "You can download Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if pip is installed
try {
    $pipVersion = python -m pip --version
    Write-Host "✓ pip detected: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: pip not found. Please ensure pip is installed." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists, create if not
if (-not (Test-Path -Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    try {
        python -m venv venv
        Write-Host "✓ Virtual environment created successfully." -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✓ Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated." -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to activate virtual environment." -ForegroundColor Red
    exit 1
}

# Fix config.py if it doesn't have LOG_LEVEL
if (Test-Path -Path "config.py") {
    $configContent = Get-Content -Path "config.py" -Raw
    if (-not ($configContent -like "*LOG_LEVEL*")) {
        Write-Host "Updating config.py with missing LOG_LEVEL parameter..." -ForegroundColor Yellow
        Add-Content -Path "config.py" -Value "`n# Logging settings`nLOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    }
}

# Install dependencies
Write-Host "Installing dependencies... (This may take a while)" -ForegroundColor Yellow
try {
    python -m pip install -r requirements.txt
    Write-Host "✓ Dependencies installed successfully." -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to install dependencies." -ForegroundColor Red
    exit 1
}

# Check if virtual audio cable is installed
Write-Host "NOTE: For capturing system audio, you need a virtual audio cable." -ForegroundColor Yellow
Write-Host "If not already installed, download and install VB-Audio Virtual Cable:" -ForegroundColor Yellow
Write-Host "https://vb-audio.com/Cable/" -ForegroundColor Yellow
Write-Host ""
Write-Host "Instructions for system audio capture:" -ForegroundColor Yellow
Write-Host "1. Install VB-Audio Virtual Cable" -ForegroundColor Yellow
Write-Host "2. Set your system output device to 'CABLE Input'" -ForegroundColor Yellow
Write-Host "3. In the app, select 'CABLE Output' as your input device" -ForegroundColor Yellow
Write-Host ""

# Ask if user wants to run the application
Write-Host "Do you want to run the application now? (Y/N)" -ForegroundColor White
$response = Read-Host
if ($response -eq 'Y' -or $response -eq 'y') {
    Write-Host "Starting the application..." -ForegroundColor Yellow
    try {
        python gui_translator.py
    } catch {
        Write-Host "ERROR: Application execution failed." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "To run the application later, use: .\venv\Scripts\python.exe gui_translator.py" -ForegroundColor Yellow
}

# Deactivate virtual environment (only if not running the app)
if ($response -ne 'Y' -and $response -ne 'y') {
    deactivate
    Write-Host "Virtual environment deactivated." -ForegroundColor Green
} 