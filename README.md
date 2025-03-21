# Real-Time Speech Translation with Speaker Identification

This application captures audio (from microphone or system audio), performs speech-to-text on spoken Russian (or other supported languages), identifies different speakers through diarization, and translates the output into English in real-time.

## Features

- **System-Level Audio Capture**: Works with virtual audio cables to capture audio from apps like Zoom, Microsoft Teams, etc.
- **Speech-to-Text Transcription**: Using OpenAI's Whisper model for accurate transcription
- **Speaker Diarization**: Identifies and labels different speakers in the conversation
- **Real-Time Translation**: Translates transcribed text to English using MarianMT
- **User-Friendly GUI**: Simple interface to display original text and translations with speaker labels

## Requirements

- Python 3.8 or newer
- NVIDIA GPU recommended for better performance (but not required)
- For system audio capture:
  - **Windows**: VB-Audio Virtual Cable
  - **macOS**: BlackHole or similar virtual audio device

## Installation

### Windows

1. Clone or download this repository
2. Run the setup script by right-clicking on `setup_and_run_windows.ps1` and selecting "Run with PowerShell"
   - If you encounter a security error, you may need to run:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
     ```
   - Then run: `.\setup_and_run_windows.ps1`

### macOS

1. Clone or download this repository
2. Open Terminal in the project directory
3. Make the setup script executable and run it:
   ```bash
   chmod +x setup_and_run_mac.sh
   ./setup_and_run_mac.sh
   ```

## Setting Up System Audio Capture

### Windows (VB-Audio Virtual Cable)

1. Download and install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
2. In Windows sound settings:
   - Set "CABLE Input" as your default output device
   - Your actual speakers/headphones will not play sound at this point
3. In the application, select "CABLE Output" as your input device
4. Optional: To hear audio while capturing, set up a listen option:
   - Right-click on "CABLE Output" in Sound Control Panel
   - Select "Properties" → "Listen" tab → Check "Listen to this device"
   - Choose your speakers/headphones under "Playback through this device"

### macOS (BlackHole)

1. Install [BlackHole](https://github.com/ExistentialAudio/BlackHole)
2. Open "Audio MIDI Setup" application
3. Create a Multi-Output Device:
   - Click the "+" button → "Create Multi-Output Device"
   - Check both your speakers/headphones and "BlackHole 2ch"
   - Set "Multi-Output Device" as your system output device
4. In the application, select "BlackHole 2ch" as your input device

## Usage

1. Launch the application using the setup script or by running:
   - Windows: `.\venv\Scripts\python.exe gui_translator.py`
   - macOS: `./venv/bin/python3 gui_translator.py`

2. Configure the application:
   - Select source language (default: Russian)
   - Select target language (default: English)
   - Choose your input device (microphone or virtual audio device)
   - Toggle speaker identification if needed

3. Click "Start Listening" to begin capturing and translating audio

4. The application will display:
   - Original transcribed text with speaker labels
   - English translations
   - Performance statistics

## Supported Languages

The application primarily supports these source languages:
- Russian (ru)
- Arabic (ar)
- Chinese (zh)
- Farsi/Persian (fa)

And translates to:
- English (en)

## Troubleshooting

- **No audio detected**: Ensure the correct input device is selected in the application
- **Poor transcription quality**: Try adjusting system volume levels
- **Performance issues**: If using a CPU-only system, try selecting a smaller Whisper model in the config.py file
- **Speaker identification not working**: Check if pyannote.audio is properly installed

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Hugging Face Transformers](https://huggingface.co/transformers/) for MarianMT translation models #   l i v e - t r a n s l a t i o n  
 