import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

st.title("🎙️ Deep Learning Voice Chatbot")
st.write("Upload an audio file or record sound to process.")

audio_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])

if audio_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_path = temp_audio.name

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
        st.subheader("1. Recognized Text:")
        st.write(text)

        # Response Logic
        response_text = f"I received your message: '{text}'"
        st.subheader("2. Chatbot Response:")
        st.write(response_text)

        # TTS Output
        tts = gTTS(text=response_text, lang='en')
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(out_path)
        st.audio(out_path)

    except Exception as e:
        st.error(f"Error processing audio: {e}")
