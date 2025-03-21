import whisper
import torch
from transformers import MarianMTModel, MarianTokenizer
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, StringVar
import datetime
import logging

# Default settings
SOURCE_LANGUAGE = 'ru'
TARGET_LANGUAGE = 'en'
WHISPER_MODEL = 'medium'
SAMPLE_RATE = 16000
BLOCK_DURATION = 3.0
USE_FP16 = True
MIN_BLOCKS_FOR_STATS = 3
USE_DIARIZATION = True
LOG_LEVEL = "INFO"
HF_TOKEN = ""

# Import configuration if available, overriding defaults
try:
    from config import *
except ImportError:
    print("Config file not found, using default settings.")

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('RealTimeTranslator')

# Import pyannote.audio for speaker diarization (if enabled)
if USE_DIARIZATION:
    try:
        from pyannote.audio import Pipeline
        DIARIZATION_AVAILABLE = True
        logger.info("Speaker diarization is enabled and package is available")
    except ImportError:
        logger.warning("pyannote.audio not found. Speaker diarization will be disabled.")
        DIARIZATION_AVAILABLE = False
else:
    DIARIZATION_AVAILABLE = False
    logger.info("Speaker diarization is disabled in config")

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Speech Translator")
        self.root.geometry("800x600")
        self.root.minsize(640, 480)
        
        # Set app style
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 12))
        style.configure('TLabel', font=('Arial', 12))
        style.configure('Stats.TLabel', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        
        # Setup variables
        self.source_lang = StringVar(value=SOURCE_LANGUAGE)
        self.target_lang = StringVar(value=TARGET_LANGUAGE)
        self.use_diarization = DIARIZATION_AVAILABLE and USE_DIARIZATION
        self.is_running = False
        self.audio_thread = None
        self.keyboard_thread = None
        self.audio_queue = queue.Queue()
        self.buffer = np.zeros((0, 1), dtype=np.float32)
        
        # Audio devices
        self.available_devices = self.get_audio_devices()
        self.selected_device = StringVar()
        if len(self.available_devices) > 0:
            self.selected_device.set(self.available_devices[0][0])  # Default to first device
        
        # Speaker tracking
        self.current_speaker = "Speaker 1"
        self.speaker_map = {}  # Maps speaker IDs to friendly names
        self.next_speaker_id = 1
        
        # Performance tracking
        self.performance_stats = {
            'transcription_times': [],
            'translation_times': [],
            'diarization_times': [],
            'total_times': []
        }
        self.utterance_counter = 0
        
        # Create UI elements
        self.create_widgets()
        
        # Initialize models
        self.initialize_thread = threading.Thread(target=self.load_models)
        self.initialize_thread.daemon = True
        self.initialize_thread.start()
    
    def get_audio_devices(self):
        """Get list of available audio input devices"""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:  # Input device
                    devices.append((f"{device['name']}", i))
            logger.info(f"Found {len(devices)} audio input devices")
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
        return devices
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=(10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Real-time Speech Translator", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Language selection
        lang_frame = ttk.LabelFrame(control_frame, text="Language Settings")
        lang_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X)
        
        ttk.Label(lang_frame, text="Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        source_combo = ttk.Combobox(lang_frame, textvariable=self.source_lang, width=10)
        source_combo['values'] = ('ru', 'ar', 'zh', 'fa')
        source_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(lang_frame, text="Target:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        target_combo = ttk.Combobox(lang_frame, textvariable=self.target_lang, width=10)
        target_combo['values'] = ('en',)
        target_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Audio device selection
        device_frame = ttk.LabelFrame(main_frame, text="Audio Device")
        device_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(device_frame, text="Input Device:").pack(side=tk.LEFT, padx=5, pady=5)
        device_combo = ttk.Combobox(device_frame, textvariable=self.selected_device, width=40)
        device_combo['values'] = [device[0] for device in self.available_devices]
        device_combo.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Diarization toggle
        if DIARIZATION_AVAILABLE:
            self.diarization_var = tk.BooleanVar(value=self.use_diarization)
            diarization_check = ttk.Checkbutton(
                device_frame, 
                text="Enable Speaker Identification", 
                variable=self.diarization_var,
                command=self.toggle_diarization
            )
            diarization_check.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(button_frame, text="Start Listening", command=self.start_listening)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_listening, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = StringVar(value="Loading models...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(fill=tk.X, pady=(0, 10))
        
        # Text display frame
        text_frame = ttk.LabelFrame(main_frame, text="Transcription & Translation")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Original text
        ttk.Label(text_frame, text="Original Speech:").pack(anchor=tk.W, padx=5)
        self.original_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=4)
        self.original_text.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        # Translated text
        ttk.Label(text_frame, text="Translation:").pack(anchor=tk.W, padx=5)
        self.translated_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=8)
        self.translated_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 10))
        
        # Performance metrics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Performance Metrics")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="No performance data yet", style='Stats.TLabel')
        self.stats_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Current processing info
        self.current_info_var = StringVar(value="")
        self.current_info = ttk.Label(main_frame, textvariable=self.current_info_var)
        self.current_info.pack(fill=tk.X)
    
    def toggle_diarization(self):
        """Toggle speaker diarization on/off"""
        self.use_diarization = self.diarization_var.get()
        logger.info(f"Speaker diarization set to: {self.use_diarization}")
    
    def load_models(self):
        try:
            # Update status
            self.update_status("Checking GPU availability...")
            
            # Check for GPU
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.update_status(f"GPU detected: {gpu_name}")
                logger.info(f"GPU detected: {gpu_name}")
            else:
                self.update_status("No GPU detected. Performance will be degraded.")
                logger.warning("No GPU detected. Performance will be degraded.")
            
            # Load MarianMT model
            self.update_status(f"Loading translation model for {self.source_lang.get()} → {self.target_lang.get()}...")
            marian_model_name = f'Helsinki-NLP/opus-mt-{self.source_lang.get()}-{self.target_lang.get()}'
            
            self.tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
            self.translator = MarianMTModel.from_pretrained(marian_model_name)
            
            if torch.cuda.is_available():
                self.translator = self.translator.cuda()
            
            # Load Whisper model
            self.update_status(f"Loading Whisper model ({WHISPER_MODEL})...")
            self.whisper_model = whisper.load_model(WHISPER_MODEL)
            
            if torch.cuda.is_available():
                self.whisper_model = self.whisper_model.cuda()
            
            # Load diarization model if enabled
            if self.use_diarization and DIARIZATION_AVAILABLE:
                self.update_status("Loading speaker diarization model...")
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization@2.1",
                        use_auth_token=HF_TOKEN if HF_TOKEN else None
                    )
                    if torch.cuda.is_available():
                        self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                    logger.info("Speaker diarization model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading diarization model: {e}")
                    self.use_diarization = False
                    if hasattr(self, 'diarization_var'):
                        self.diarization_var.set(False)
                    messagebox.showwarning(
                        "Diarization Error", 
                        f"Could not load speaker diarization model: {str(e)}\n\nDisabling speaker diarization."
                    )
            
            self.update_status("Ready to start! Click 'Start Listening' to begin.")
            self.root.after(0, lambda: self.start_button.configure(state=tk.NORMAL))
            
        except Exception as e:
            error_msg = f"Error initializing models: {str(e)}"
            logger.error(error_msg)
            self.update_status(error_msg)
            messagebox.showerror("Initialization Error", error_msg)
    
    def update_status(self, message):
        def _update():
            self.status_var.set(message)
        self.root.after(0, _update)
        logger.info(message)
    
    def update_stats(self):
        if len(self.performance_stats['total_times']) < MIN_BLOCKS_FOR_STATS:
            stats_text = "Collecting performance data..."
        else:
            # Calculate stats
            avg_transcribe = np.mean(self.performance_stats['transcription_times'])
            avg_translate = np.mean(self.performance_stats['translation_times'])
            avg_total = np.mean(self.performance_stats['total_times'])
            
            # Add diarization stats if used
            diarization_text = ""
            if self.use_diarization and len(self.performance_stats['diarization_times']) > 0:
                avg_diarize = np.mean(self.performance_stats['diarization_times'])
                diarization_text = f"Diarization: {avg_diarize:.3f}s | "
            
            # Recent stats (last 10 or fewer)
            recent_count = min(10, len(self.performance_stats['total_times']))
            recent_transcribe = np.mean(self.performance_stats['transcription_times'][-recent_count:])
            recent_translate = np.mean(self.performance_stats['translation_times'][-recent_count:])
            recent_total = np.mean(self.performance_stats['total_times'][-recent_count:])
            
            recent_diarization_text = ""
            if self.use_diarization and len(self.performance_stats['diarization_times']) > 0:
                recent_diarize = np.mean(self.performance_stats['diarization_times'][-recent_count:])
                recent_diarization_text = f"Diarization: {recent_diarize:.3f}s | "
            
            stats_text = (
                f"Recent ({recent_count} utterances): "
                f"Transcription: {recent_transcribe:.3f}s | "
                f"Translation: {recent_translate:.3f}s | "
                f"{recent_diarization_text}"
                f"Total: {recent_total:.3f}s | "
                f"Rate: {1/recent_total:.2f} utt/sec\n"
                f"Overall ({len(self.performance_stats['total_times'])} utterances): "
                f"Transcription: {avg_transcribe:.3f}s | "
                f"Translation: {avg_translate:.3f}s | "
                f"{diarization_text}"
                f"Total: {avg_total:.3f}s"
            )
        
        def _update():
            self.stats_label.configure(text=stats_text)
        self.root.after(0, _update)
    
    def get_device_id(self):
        """Get the selected device ID from the dropdown"""
        selected = self.selected_device.get()
        for device_name, device_id in self.available_devices:
            if device_name == selected:
                return device_id
        return None  # Default device
    
    def start_listening(self):
        self.is_running = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        
        # Reset audio buffer and speaker tracking
        self.buffer = np.zeros((0, 1), dtype=np.float32)
        if self.use_diarization:
            self.speaker_map = {}
            self.next_speaker_id = 1
        
        # Start audio thread
        self.audio_thread = threading.Thread(target=self.audio_processing_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        self.update_status("Listening... Speak into the microphone.")
    
    def stop_listening(self):
        self.is_running = False
        self.stop_button.configure(state=tk.DISABLED)
        
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        
        self.start_button.configure(state=tk.NORMAL)
        self.update_status("Stopped listening. Click 'Start Listening' to begin again.")
        self.update_stats()
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())
    
    def process_diarization(self, audio_np):
        """Process audio for speaker diarization"""
        start_time = time.time()
        
        # Save temporary audio file for diarization
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Write audio to temp file
        sf.write(temp_filename, audio_np, SAMPLE_RATE)
        
        # Run diarization
        try:
            diarization = self.diarization_pipeline(temp_filename)
            
            # Find the dominant speaker in this segment
            max_duration = 0
            dominant_speaker = None
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                if duration > max_duration:
                    max_duration = duration
                    dominant_speaker = speaker
            
            # Map to friendly speaker name
            if dominant_speaker not in self.speaker_map:
                self.speaker_map[dominant_speaker] = f"Speaker {self.next_speaker_id}"
                self.next_speaker_id += 1
            
            speaker_name = self.speaker_map[dominant_speaker]
            
        except Exception as e:
            logger.error(f"Diarization error: {e}")
            speaker_name = "Unknown Speaker"
        
        # Clean up temp file
        import os
        try:
            os.unlink(temp_filename)
        except:
            pass
        
        diarization_time = time.time() - start_time
        self.performance_stats['diarization_times'].append(diarization_time)
        
        return speaker_name, diarization_time
    
    def translate_text(self, text):
        start_time = time.time()
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            
        translated = self.translator.generate(**inputs)
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        
        translation_time = time.time() - start_time
        self.performance_stats['translation_times'].append(translation_time)
        
        return translated_text, translation_time
    
    def update_text_display(self, speaker, original, translated):
        def _update():
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Update original text
            self.original_text.config(state=tk.NORMAL)
            self.original_text.insert(tk.END, f"[{timestamp}] {speaker}: {original}\n\n")
            self.original_text.see(tk.END)
            self.original_text.config(state=tk.DISABLED)
            
            # Update translated text
            self.translated_text.config(state=tk.NORMAL)
            self.translated_text.insert(tk.END, f"[{timestamp}] {speaker}: {translated}\n\n")
            self.translated_text.see(tk.END)
            self.translated_text.config(state=tk.DISABLED)
        
        self.root.after(0, _update)
    
    def update_current_info(self, transcription_time, translation_time, diarization_time, total_time):
        def _update():
            diarization_text = f"Diarization: {diarization_time:.3f}s | " if diarization_time is not None else ""
            
            info_text = (
                f"Current utterance: "
                f"Transcription: {transcription_time:.3f}s | "
                f"Translation: {translation_time:.3f}s | "
                f"{diarization_text}"
                f"Total: {total_time:.3f}s | "
                f"Utterances processed: {self.utterance_counter}"
            )
            self.current_info_var.set(info_text)
        
        self.root.after(0, _update)
    
    def audio_processing_loop(self):
        try:
            device_id = self.get_device_id()
            with sd.InputStream(
                samplerate=SAMPLE_RATE, 
                channels=1, 
                callback=self.audio_callback,
                blocksize=int(SAMPLE_RATE * 0.5),
                device=device_id
            ):
                logger.info(f"Started audio input stream with device ID: {device_id}")
                
                while self.is_running:
                    if not self.audio_queue.empty():
                        data = self.audio_queue.get()
                        self.buffer = np.concatenate((self.buffer, data))
                        
                        # Process in blocks
                        if len(self.buffer) > BLOCK_DURATION * SAMPLE_RATE:
                            total_start_time = time.time()
                            
                            audio_block = self.buffer[:int(BLOCK_DURATION * SAMPLE_RATE)]
                            self.buffer = self.buffer[int(BLOCK_DURATION * SAMPLE_RATE):]
                            
                            audio_np = audio_block.flatten()
                            
                            # Whisper ASR
                            transcribe_start_time = time.time()
                            audio_tensor = torch.from_numpy(audio_np).float()
                            if torch.cuda.is_available():
                                audio_tensor = audio_tensor.cuda()
                                
                            result = self.whisper_model.transcribe(
                                audio_tensor, 
                                language=self.source_lang.get(),
                                fp16=USE_FP16
                            )
                            
                            transcription_time = time.time() - transcribe_start_time
                            self.performance_stats['transcription_times'].append(transcription_time)
                            
                            original_text = result["text"].strip()
                            if original_text:
                                self.utterance_counter += 1
                                
                                # Speaker diarization (if enabled)
                                diarization_time = None
                                if self.use_diarization:
                                    speaker, diarization_time = self.process_diarization(audio_np)
                                else:
                                    speaker = "Speaker"
                                
                                # Translation
                                translated_text, translation_time = self.translate_text(original_text)
                                
                                # Calculate total time
                                total_time = time.time() - total_start_time
                                self.performance_stats['total_times'].append(total_time)
                                
                                # Update UI
                                self.update_text_display(speaker, original_text, translated_text)
                                self.update_current_info(transcription_time, translation_time, diarization_time, total_time)
                                self.update_stats()
                                
                                # Log the result
                                logger.info(f"{speaker}: {original_text} → {translated_text}")
                    
                    # Small sleep to prevent CPU overload
                    time.sleep(0.01)
                    
        except Exception as e:
            error_msg = f"Error during audio processing: {str(e)}"
            logger.error(error_msg)
            self.update_status(error_msg)
            messagebox.showerror("Processing Error", error_msg)
            self.stop_listening()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop() 