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

load_dotenv()

try:
    genai.configure(api_key=os.getenv('GEMINI_API'))
except Exception as e:
    print("Error occured",e)

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
            history=[
                {"role": "user", "parts": "Hello"},
                {"role": "model", "parts": "Great to meet you. What would you like to know?"},
            ]
        )
        response = chat.send_message(prompt)
        return response.text

    except Exception as e:
        return f"Error: {e}"
    
#--> This will use to generate prompt
def gen_prompt(text, response_type="Friendly and interactive", *args):
    return f"{text}. Response type:- {response_type}. {args}"

def play_audio(text):
    tts = gTTS(text)
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)

    # Convert to NumPy array for sounddevice
    audio_data = BytesIO(audio_file.read()).getvalue()
    audio_segment = AudioSegment.from_file(BytesIO(audio_data))
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0

    # Play audio
    sd.play(samples, samplerate=audio_segment.frame_rate)
    sd.wait()