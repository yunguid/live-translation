"""
# Configuration settings for Real-time Speech Translator

# Language settings
SOURCE_LANGUAGE = 'ru'  # Source language code (ru for Russian)
TARGET_LANGUAGE = 'en'  # Target language code (en for English)

# Whisper ASR model settings
WHISPER_MODEL = 'medium'  # Options: tiny, base, small, medium, large
USE_FP16 = True  # Use half-precision floating point (faster on modern GPUs)

# Audio processing settings
SAMPLE_RATE = 16000  # Sample rate in Hz
BLOCK_DURATION = 3.0  # Duration of audio blocks to process in seconds

# Performance settings
MIN_BLOCKS_FOR_STATS = 3  # Minimum number of blocks before showing performance stats

# Speaker diarization settings
USE_DIARIZATION = True  # Enable or disable speaker diarization
HF_TOKEN = ""  # Your HuggingFace token for pyannote.audio (leave empty to use local models)

# Logging settings
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
""" 