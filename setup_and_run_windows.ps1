#!/usr/bin/env pwsh
# Real-time Translation App Setup and Run Script for Windows
# ==========================================================

# Set error action preference
$ErrorActionPreference = "Stop"

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output "$args"
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# Display banner
Write-ColorOutput Green "======================================================"
Write-ColorOutput Green "  Real-time Translation App - Windows Setup and Run   "
Write-ColorOutput Green "======================================================"
Write-ColorOutput White ""

# Check if Python is installed
$pythonVersion = & python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Python not found. Please install Python 3.8 or newer."
    Write-ColorOutput Yellow "You can download Python from https://www.python.org/downloads/"
    exit 1
}
Write-ColorOutput Green "✓ Python detected: $pythonVersion"

# Check if pip is installed
$pipVersion = & python -m pip --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: pip not found. Please ensure pip is installed."
    exit 1
}
Write-ColorOutput Green "✓ pip detected: $pipVersion"

# Check if virtual environment exists, create if not
if (-not (Test-Path -Path "venv")) {
    Write-ColorOutput Yellow "Creating virtual environment..."
    & python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "ERROR: Failed to create virtual environment."
        exit 1
    }
    Write-ColorOutput Green "✓ Virtual environment created successfully."
} else {
    Write-ColorOutput Green "✓ Virtual environment already exists."
}

# Activate virtual environment
Write-ColorOutput Yellow "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Failed to activate virtual environment."
    exit 1
}
Write-ColorOutput Green "✓ Virtual environment activated."

# Install dependencies
Write-ColorOutput Yellow "Installing dependencies... (This may take a while)"
& python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Failed to install dependencies."
    exit 1
}
Write-ColorOutput Green "✓ Dependencies installed successfully."

# Check if virtual audio cable is installed
Write-ColorOutput Yellow "NOTE: For capturing system audio, you need a virtual audio cable."
Write-ColorOutput Yellow "If not already installed, download and install VB-Audio Virtual Cable:"
Write-ColorOutput Yellow "https://vb-audio.com/Cable/"
Write-ColorOutput Yellow ""
Write-ColorOutput Yellow "Instructions for system audio capture:"
Write-ColorOutput Yellow "1. Install VB-Audio Virtual Cable"
Write-ColorOutput Yellow "2. Set your system output device to 'CABLE Input'"
Write-ColorOutput Yellow "3. In the app, select 'CABLE Output' as your input device"
Write-ColorOutput Yellow ""

# Ask if user wants to run the application
Write-ColorOutput White "Do you want to run the application now? (Y/N)"
$response = Read-Host
if ($response -eq 'Y' -or $response -eq 'y') {
    Write-ColorOutput Yellow "Starting the application..."
    & python gui_translator.py
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "ERROR: Application execution failed."
        exit 1
    }
} else {
    Write-ColorOutput Yellow "To run the application later, use: .\venv\Scripts\python.exe gui_translator.py"
}

# Deactivate virtual environment (only if not running the app)
if ($response -ne 'Y' -and $response -ne 'y') {
    & deactivate
    Write-ColorOutput Green "Virtual environment deactivated."
} 