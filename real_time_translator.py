import whisper
import torch
from transformers import MarianMTModel, MarianTokenizer
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from pynput import keyboard
import os
import datetime
import sys
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box

# Setup Rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
log = logging.getLogger("real_time_translator")

# Import configuration
try:
    from config import *
    log.info("Loaded configuration from config.py")
except ImportError:
    log.warning("Config file not found. Using default settings.")
    # Default settings
    SOURCE_LANGUAGE = 'ru'
    TARGET_LANGUAGE = 'en'
    WHISPER_MODEL = 'medium'
    SAMPLE_RATE = 16000
    BLOCK_DURATION = 3.0
    CLEAR_TERMINAL = True
    DISPLAY_INTERVAL = 10
    USE_FP16 = True
    MIN_BLOCKS_FOR_STATS = 3

# --------------------------- SETTINGS --------------------------- #
# Use settings from config file
source_lang = SOURCE_LANGUAGE
translate_to = TARGET_LANGUAGE
whisper_model_name = WHISPER_MODEL
sample_rate = SAMPLE_RATE
block_duration = BLOCK_DURATION

# MarianMT model name
marian_model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{translate_to}'

# Performance tracking
performance_stats = {
    'transcription_times': [],
    'translation_times': [],
    'total_times': [],
    'audio_durations': []
}

# ---------------------- System Check ---------------------- #
def check_gpu_compatibility():
    """Verify GPU compatibility and configuration for deep learning tasks"""
    if not torch.cuda.is_available():
        log.error("❌ No CUDA-compatible GPU detected. This application requires a GPU for optimal performance.")
        log.warning("The program will continue but performance will be significantly degraded.")
        return False
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
    
    # Create a table for GPU information
    table = Table(title="GPU Information", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("GPU Name", gpu_name)
    table.add_row("GPU Count", str(gpu_count))
    table.add_row("Current Device", str(current_device))
    table.add_row("Total Memory", f"{gpu_memory:.2f} GB")
    table.add_row("CUDA Version", torch.version.cuda or "Unknown")
    
    # Check for CUDNN (important for performance)
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_enabled = torch.backends.cudnn.enabled
    table.add_row("cuDNN Available", "✅ Yes" if cudnn_available else "❌ No")
    table.add_row("cuDNN Enabled", "✅ Yes" if cudnn_enabled else "❌ No")
    
    # Verify compute capability (for half-precision support)
    compute_capability = torch.cuda.get_device_capability(current_device)
    cc_version = f"{compute_capability[0]}.{compute_capability[1]}"
    fp16_supported = compute_capability >= (5, 3)
    table.add_row("Compute Capability", cc_version)
    table.add_row("FP16 Support", "✅ Yes" if fp16_supported else "❌ No")
    
    # Display the table
    console.print(table)
    
    # Configure based on GPU detection
    if not fp16_supported and USE_FP16:
        log.warning("GPU does not support FP16 but FP16 is enabled in config. Disabling FP16.")
        global USE_FP16
        USE_FP16 = False
    
    # Optimize CUDNN if available
    if cudnn_available:
        torch.backends.cudnn.benchmark = True
        log.info("cuDNN benchmark mode enabled for faster performance")
    
    return True

def check_system():
    """Verify system compatibility and report status"""
    console.rule("[bold blue]System Compatibility Check")
    
    # Check Python version
    python_version = sys.version.split()[0]
    log.info(f"Python version: {python_version}")
    
    # Check PyTorch version
    log.info(f"PyTorch version: {torch.__version__}")
    
    # Check GPU
    gpu_available = check_gpu_compatibility()
    
    # Check if MarianMT model exists
    try:
        with console.status("[bold green]Checking translation model availability..."):
            log.info(f"Checking if translation model exists for {source_lang} → {translate_to}...")
            tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
            log.info(f"✅ Translation model found: {marian_model_name}")
    except Exception as e:
        log.error(f"❌ Translation model not found for {source_lang} → {translate_to}")
        log.error(f"Error details: {str(e)}")
        log.info("Please check available models at: https://huggingface.co/Helsinki-NLP")
        sys.exit(1)
    
    # Check audio device
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        log.info(f"Default audio input device: {default_input['name']}")
    except Exception as e:
        log.error(f"❌ Error querying audio devices: {str(e)}")
        log.warning("Audio capture may not work properly.")
    
    return gpu_available

# Call system check
gpu_available = check_system()

# ---------------------- Load Models ---------------------- #
def load_whisper_model():
    """Load and configure Whisper model with progress reporting"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Loading Whisper model..."),
        console=console
    ) as progress:
        task = progress.add_task("Loading...", total=None)
        
        start_time = time.time()
        whisper_model = whisper.load_model(whisper_model_name)
        
        if gpu_available:
            whisper_model = whisper_model.cuda()
            # Force a CUDA kernel launch to verify GPU works
            dummy_input = torch.zeros(1, 1).cuda()
            dummy_output = dummy_input * 2
            torch.cuda.synchronize()  # Ensure operation is complete
            
        load_time = time.time() - start_time
        progress.update(task, completed=100)
    
    # Memory usage info
    if gpu_available:
        allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved_memory = torch.cuda.memory_reserved() / (1024**3)    # GB
        log.info(f"Whisper model '{whisper_model_name}' loaded on GPU in {load_time:.2f}s")
        log.info(f"GPU memory: {allocated_memory:.2f} GB allocated, {reserved_memory:.2f} GB reserved")
    else:
        log.info(f"Whisper model '{whisper_model_name}' loaded on CPU in {load_time:.2f}s")
        
    return whisper_model

def load_marian_model():
    """Load and configure MarianMT model with progress reporting"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Loading MarianMT model..."),
        console=console
    ) as progress:
        task = progress.add_task("Loading...", total=None)
        
        start_time = time.time()
        tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
        translator = MarianMTModel.from_pretrained(marian_model_name)
        
        if gpu_available:
            translator = translator.cuda()
            # Verify model is on GPU by running a quick forward pass
            dummy_input = tokenizer("Test", return_tensors="pt").to('cuda')
            with torch.no_grad():
                translator.generate(**dummy_input)
            torch.cuda.synchronize()
            
        load_time = time.time() - start_time
        progress.update(task, completed=100)
    
    # Memory usage after loading
    if gpu_available:
        allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        log.info(f"MarianMT model loaded on GPU in {load_time:.2f}s")
        log.info(f"Total GPU memory allocated: {allocated_memory:.2f} GB")
    else:
        log.info(f"MarianMT model loaded on CPU in {load_time:.2f}s")
        
    return tokenizer, translator

console.rule("[bold blue]Loading Models")
whisper_model = load_whisper_model()
tokenizer, translator = load_marian_model()

# ---------------------- Audio Recording ------------------------- #
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        log.warning(f"Audio callback status: {status}")
    audio_queue.put(indata.copy())

def listen_audio():
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback, 
                           blocksize=int(sample_rate * 0.5)):  # Process in 0.5s blocks
            log.info("🎤 Microphone initialized successfully")
            while running:
                sd.sleep(100)
    except Exception as e:
        log.error(f"❌ Error initializing microphone: {str(e)}")
        log.error("Please check your microphone settings and restart the program.")
        global running
        running = False

# ------------------------ Translate Function ------------------------ #
def translate_text(text):
    """Translate text with performance monitoring"""
    start_time = time.time()
    
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    if gpu_available:
        inputs = inputs.to('cuda')
    
    # Generate translation with GPU memory tracking
    if gpu_available:
        before_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    with torch.no_grad():  # Disable gradient calculation for inference
        translated = translator.generate(**inputs, max_length=100)
    
    if gpu_available:
        torch.cuda.synchronize()  # Make sure generation is complete
        after_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        memory_used = after_memory - before_memory
        if memory_used > 0:
            log.debug(f"Translation used {memory_used:.2f} MB of GPU memory")
    
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    translation_time = time.time() - start_time
    performance_stats['translation_times'].append(translation_time)
    
    return translated_text, translation_time

# ------------------------ Clear Terminal ------------------------ #
def clear_terminal():
    if CLEAR_TERMINAL:
        os.system('cls' if os.name == 'nt' else 'clear')

# ------------------ Performance Reporting ----------------------- #
def create_performance_table():
    """Create a rich table with performance metrics"""
    if len(performance_stats['total_times']) < MIN_BLOCKS_FOR_STATS:
        return Panel("Collecting performance data...", title="Performance Metrics")
    
    # Calculate statistics
    avg_transcribe = np.mean(performance_stats['transcription_times'])
    avg_translate = np.mean(performance_stats['translation_times'])
    avg_total = np.mean(performance_stats['total_times'])
    
    # Recent statistics
    recent_count = min(10, len(performance_stats['total_times']))
    recent_transcribe = np.mean(performance_stats['transcription_times'][-recent_count:])
    recent_translate = np.mean(performance_stats['translation_times'][-recent_count:])
    recent_total = np.mean(performance_stats['total_times'][-recent_count:])
    
    # Calculate real-time factor if audio durations are available
    rtf = "N/A"
    if performance_stats['audio_durations']:
        avg_duration = np.mean(performance_stats['audio_durations'])
        if avg_duration > 0:
            rtf = f"{avg_total/avg_duration:.2f}x"
    
    # Create performance table
    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Recent", style="green")
    table.add_column("Overall", style="blue")
    
    table.add_row("Transcription Time", f"{recent_transcribe:.3f}s", f"{avg_transcribe:.3f}s")
    table.add_row("Translation Time", f"{recent_translate:.3f}s", f"{avg_translate:.3f}s")
    table.add_row("Total Processing", f"{recent_total:.3f}s", f"{avg_total:.3f}s")
    table.add_row("Processing Rate", f"{1/recent_total:.2f} utt/sec", f"{1/avg_total:.2f} utt/sec")
    table.add_row("Real-time Factor", rtf, "")
    
    # Add GPU utilization if available
    if gpu_available:
        mem_percent = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        table.add_row("GPU Memory Used", f"{mem_percent:.1f}%", "")
    
    return table

def display_performance_stats():
    table = create_performance_table()
    return table

# ------------------- Main Processing Loop ----------------------- #
running = True

def on_press(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        log.info("Stopping the program...")

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()

audio_thread = threading.Thread(target=listen_audio)
audio_thread.daemon = True
audio_thread.start()

console.rule("[bold blue]Real-time Speech Translation System")
console.print(f"🔊 Source language: [bold]{source_lang}[/bold] → [bold]{translate_to}[/bold]")
console.print("🔍 Press [bold red]ESC[/bold] to stop and view final performance stats.\n")

buffer = np.zeros((0, 1), dtype=np.float32)
last_update_time = time.time()
utterance_counter = 0

try:
    with Live(Panel("Waiting for speech...", title="Status"), refresh_per_second=4) as live:
        while running:
            if not audio_queue.empty():
                data = audio_queue.get()
                buffer = np.concatenate((buffer, data))

                # Process in blocks
                if len(buffer) > block_duration * sample_rate:
                    total_start_time = time.time()
                    
                    # Update live display to processing
                    live.update(Panel("Processing audio...", title="Status"))
                    
                    audio_block = buffer[:int(block_duration * sample_rate)]
                    buffer = buffer[int(block_duration * sample_rate):]

                    audio_np = audio_block.flatten()
                    performance_stats['audio_durations'].append(block_duration)

                    # Whisper ASR
                    log.info(f"Starting transcription of {block_duration:.1f}s audio block...")
                    transcribe_start_time = time.time()
                    
                    audio_tensor = torch.from_numpy(audio_np).float()
                    if gpu_available:
                        audio_tensor = audio_tensor.cuda()
                        
                    # Monitor GPU memory before transcription
                    if gpu_available:
                        before_mem = torch.cuda.memory_allocated() / (1024**2)  # MB
                    
                    result = whisper_model.transcribe(audio_tensor, language=source_lang, fp16=USE_FP16)
                    
                    # Verify GPU was used and track memory
                    if gpu_available:
                        torch.cuda.synchronize()
                        after_mem = torch.cuda.memory_allocated() / (1024**2)  # MB
                        mem_used = after_mem - before_mem
                        log.debug(f"Transcription used {mem_used:.2f} MB of GPU memory")
                    
                    transcription_time = time.time() - transcribe_start_time
                    performance_stats['transcription_times'].append(transcription_time)
                    log.info(f"Transcription completed in {transcription_time:.3f}s")

                    original_text = result["text"].strip()
                    if original_text:
                        utterance_counter += 1
                        log.info(f"Detected speech in {source_lang}: '{original_text}'")
                        
                        # Translation via MarianMT
                        log.info("Starting translation...")
                        translated_text, translation_time = translate_text(original_text)
                        log.info(f"Translation completed in {translation_time:.3f}s")
                        
                        total_time = time.time() - total_start_time
                        performance_stats['total_times'].append(total_time)
                        
                        # Format timestamp
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        
                        # Create display output
                        result_panel = Panel(
                            f"[bold cyan]Original ({source_lang}):[/bold cyan]\n{original_text}\n\n"
                            f"[bold green]Translation ({translate_to}):[/bold green]\n{translated_text}",
                            title=f"Translation Result ({timestamp})"
                        )
                        
                        # Processing info
                        info_text = (
                            f"[bold]Processing Times:[/bold]\n"
                            f"  Transcription: {transcription_time:.3f}s | "
                            f"Translation: {translation_time:.3f}s | "
                            f"Total: {total_time:.3f}s | "
                            f"Utterances: {utterance_counter}"
                        )
                        
                        # Combine panels
                        combined_display = Panel.fit(
                            result_panel, 
                            title="Real-time Speech Translation"
                        )
                        
                        # Update display
                        live.update(combined_display)
                        
                        # Display performance stats periodically
                        current_time = time.time()
                        if current_time - last_update_time > DISPLAY_INTERVAL:
                            stats_table = display_performance_stats()
                            live.update(Panel(stats_table, title="Performance Metrics"))
                            time.sleep(2)  # Show stats for 2 seconds
                            live.update(combined_display)  # Return to translation display
                            last_update_time = current_time
                    else:
                        live.update(Panel("No speech detected in audio block", title="Status"))

            # Small sleep to prevent CPU overload
            time.sleep(0.01)

except KeyboardInterrupt:
    running = False

except Exception as e:
    log.exception(f"An error occurred during processing")
    running = False

finally:
    keyboard_listener.stop()
    audio_thread.join()
    
    # Display final performance stats
    if performance_stats['total_times']:
        console.rule("[bold red]FINAL PERFORMANCE STATS")
        console.print(display_performance_stats())
        
        # Export stats to CSV if needed
        # import pandas as pd
        # df = pd.DataFrame({
        #     'transcription_time': performance_stats['transcription_times'],
        #     'translation_time': performance_stats['translation_times'],
        #     'total_time': performance_stats['total_times']
        # })
        # df.to_csv('performance_stats.csv', index=False)
        # console.print("Performance stats saved to performance_stats.csv")
    
    # Show GPU memory usage summary
    if gpu_available:
        console.print(f"\nPeak GPU memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        torch.cuda.empty_cache()
        console.print("GPU memory cache cleared")
    
    console.rule("[bold blue]Program terminated") 