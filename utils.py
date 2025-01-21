import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import json
import re
import os
from numba import njit,cuda

load_dotenv()

HISTORY=[
    {"role": "user", "parts": "Hello"},
    {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    {"role": "user", "parts": "You are my english spoken friend to help me speak english. Make short conversations."},
    {"role": "model", "parts": "Ok, Great. What topic should we choose?"},
]

try:
    genai.configure(api_key=os.getenv('GEMINI_API'))
except Exception as e:
    print("Error occured",e)

def set_history(history=HISTORY):
    with open('history.json','w') as f:
        json.dump(obj=history,fp=f,indent=4)

set_history()

def get_history():
    with open('history.json','r') as f:
        history=json.load(f)
    return history

def update_history(recent_convo:dict):
    try:
        with open('history.json', 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    history.append(recent_convo)
    with open('history.json', 'w') as f:
        json.dump(history, f, indent=4)

# Speech-to-Text Function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak into the microphone.")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand your speech."
        except sr.RequestError as e:
            return f"Could not request results; {e}"
        except Exception as e:
            return f"Error: {e}"

# Text-to-Speech Function
def text_to_speech(text):
    tts = gTTS(text)
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

# Gemini (PaLM API) Call
def query_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(
            history=get_history()
        )
        response = chat.send_message(prompt)
        return response.text

    except Exception as e:
        return f"Error: {e}"

def clean_text(sentence):
    """
    Cleans the given text by removing unwanted characters and normalizing spaces.
    
    Parameters:
        sentence (str): The text to clean.
    
    Returns:
        str: The cleaned text.
    """
    cleaned_text = re.sub(r"\\|[*]", "", sentence)
    cleaned_text = re.sub(r"[\n\t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# def play_audio(text):
#     tts = gTTS(text)
#     audio_file = BytesIO()
#     tts.write_to_fp(audio_file)
#     audio_file.seek(0)

#     # Convert to NumPy array for sounddevice
#     audio_data = BytesIO(audio_file.read()).getvalue()
#     audio_segment = AudioSegment.from_file(BytesIO(audio_data))
#     samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0

#     # Play audio
#     sd.play(samples, samplerate=audio_segment.frame_rate)
#     sd.wait()

def delete_history():
    """
    Deletes the history.json file if it exists.
    """
    if os.path.exists("history.json"):
        os.remove("history.json")
        print("Chat history deleted.")
    else:
        print("No chat history found to delete.")


@njit
def resample_audio_cpu(input_audio, scale_factor):
    """
    Resamples the audio to adjust playback speed using linear interpolation.
    This version runs entirely on the CPU and is optimized with Numba.
    """
    input_length = len(input_audio)
    output_length = int(input_length / scale_factor)
    output_audio = np.zeros(output_length, dtype=np.float32)
    
    for i in range(output_length):
        input_idx = i * scale_factor
        lower = int(np.floor(input_idx))
        upper = min(lower + 1, input_length - 1)
        weight = input_idx - lower
        output_audio[i] = (1 - weight) * input_audio[lower] + weight * input_audio[upper]
    
    return output_audio

@cuda.jit
def resample_audio_cuda(input_audio, output_audio, scale_factor, input_length, output_length):
    """
    CUDA kernel to resample audio.
    """
    idx = cuda.grid(1)  # Get thread index
    if idx < output_length:
        # Map output index to input index
        input_idx = idx * scale_factor
        lower = int(input_idx)
        upper = min(lower + 1, input_length - 1)
        weight = input_idx - lower
        output_audio[idx] = input_audio[lower] * (1 - weight) + input_audio[upper] * weight

def resample_audio_with_cuda(input_audio, scale_factor):
    """
    Wrapper to use the CUDA kernel for resampling.
    """
    input_length = len(input_audio)
    output_length = int(input_length / scale_factor)
    
    # Allocate device memory
    d_input = cuda.to_device(input_audio)
    d_output = cuda.device_array(output_length, dtype=np.float32)
    
    # Launch CUDA kernel
    threads_per_block = 256
    blocks_per_grid = (output_length + threads_per_block - 1) // threads_per_block
    resample_audio_cuda[blocks_per_grid, threads_per_block](d_input, d_output, scale_factor, input_length, output_length)
    
    # Copy data back to host
    return d_output.copy_to_host()

def play_audio_streamed(text, playback_speed=1.07, chunk_size=2000):
    """
    Plays TTS audio in a streamed manner by processing text in chunks.
    """
    # Split text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        # Generate TTS audio
        tts = gTTS(chunk)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)

        # Load audio data
        audio_segment = AudioSegment.from_file(audio_file, format="mp3")
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        sample_rate = audio_segment.frame_rate

        # Resample audio for playback speed adjustment
        resampled_samples = resample_audio_with_cuda(samples, playback_speed)

        # Play the audio chunk
        sd.play(resampled_samples, samplerate=int(sample_rate * playback_speed))
        sd.wait()