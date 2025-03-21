#!/bin/bash
# Real-time Translation App Setup and Run Script for macOS
# =======================================================

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Display banner
echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  Real-time Translation App - macOS Setup and Run     ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python not found. Please install Python 3.8 or newer.${NC}"
    echo -e "${YELLOW}You can download Python from https://www.python.org/downloads/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python detected: $(python3 --version)${NC}"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}ERROR: pip not found. Please ensure pip is installed.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ pip detected: $(pip3 --version)${NC}"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create virtual environment.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Virtual environment created successfully.${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment activated.${NC}"

# Install dependencies
echo -e "${YELLOW}Installing dependencies... (This may take a while)${NC}"
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install dependencies.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dependencies installed successfully.${NC}"

# Check if virtual audio cable is installed
echo -e "${YELLOW}NOTE: For capturing system audio, you need a virtual audio device.${NC}"
echo -e "${YELLOW}If not already installed, download and install BlackHole:${NC}"
echo -e "${YELLOW}https://github.com/ExistentialAudio/BlackHole${NC}"
echo -e ""
echo -e "${YELLOW}Instructions for system audio capture:${NC}"
echo -e "${YELLOW}1. Install BlackHole${NC}"
echo -e "${YELLOW}2. Set up a Multi-Output Device in Audio MIDI Setup${NC}"
echo -e "${YELLOW}   - Include both your speakers and BlackHole${NC}"
echo -e "${YELLOW}3. Set the Multi-Output Device as your system output${NC}"
echo -e "${YELLOW}4. In the app, select 'BlackHole 2ch' as your input device${NC}"
echo -e ""

# Ask if user wants to run the application
echo -e "Do you want to run the application now? (Y/N)"
read response
if [[ $response == "Y" || $response == "y" ]]; then
    echo -e "${YELLOW}Starting the application...${NC}"
    python3 gui_translator.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Application execution failed.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}To run the application later, use: ./venv/bin/python3 gui_translator.py${NC}"
fi

# Deactivate virtual environment (only if not running the app)
if [[ $response != "Y" && $response != "y" ]]; then
    deactivate
    echo -e "${GREEN}Virtual environment deactivated.${NC}"
fi 