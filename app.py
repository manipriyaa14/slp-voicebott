import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

# Set Streamlit Page Configuration
st.set_page_config(page_title="Deep Learning Voice Chatbot", page_icon="🎙️")

# ==========================================
# 1. DEEP LEARNING MODEL DEFINITION
# ==========================================
class IntentClassifier(nn.Module):
    """Deep Learning Neural Network for Intent Classification"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Sample Intent Dataset
INTENTS = {
    "greeting": {
        "keywords": ["hello", "hi", "hey", "greetings", "morning"],
        "responses": ["Hello! How can I assist you today?", "Hi there! How can I help?"]
    },
    "goodbye": {
        "keywords": ["bye", "goodbye", "see you", "farewell", "quit"],
        "responses": ["Goodbye! Have a great day ahead.", "Bye! Feel free to talk to me anytime."]
    },
    "identity": {
        "keywords": ["who", "are", "you", "your", "name", "bot"],
        "responses": ["I am an AI Voice Chatbot powered by Deep Learning and Speech Recognition!"]
    },
    "capabilities": {
        "keywords": ["what", "can", "you", "do", "features", "help"],
        "responses": ["I can listen to your voice commands, classify your intent using Neural Networks, and reply with audio!"]
    }
}

VOCAB = sorted(list(set([kw for intent in INTENTS.values() for kw in intent["keywords"]])))
INTENT_LABELS = list(INTENTS.keys())

def bag_of_words(text, vocab):
    tokens = text.lower().split()
    return np.array([1.0 if word in tokens else 0.0 for word in vocab], dtype=np.float32)

@st.cache_resource
def load_model():
    model = IntentClassifier(len(VOCAB), 64, len(INTENT_LABELS))
    model.eval()
    return model

model = load_model()

def predict_intent(text):
    bow = torch.tensor(bag_of_words(text, VOCAB)).unsqueeze(0)
    with torch.no_grad():
        outputs = model(bow)
        _, predicted = torch.max(outputs, 1)
        intent = INTENT_LABELS[predicted.item()]
    
    matched = False
    for label, data in INTENTS.items():
        if any(kw in text.lower() for kw in data["keywords"]):
            intent = label
            matched = True
            break
            
    response = np.random.choice(INTENTS[intent]["responses"]) if matched else "I heard you, but I'm not sure how to respond to that yet."
    return intent, response

# ==========================================
# 2. STREAMLIT UI & MICROPHONE INPUT
# ==========================================
st.title("🎙️ Voice-Enabled Chatbot")
st.write("Click the microphone below to record your voice live.")

# Live Microphone Recording Widget
audio_data = st.audio_input("Record your message")

if audio_data is not None:
    # Save recorded microphone audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data.getbuffer())
        temp_audio_path = temp_audio.name

    st.subheader("1. Recorded Audio")
    st.audio(temp_audio_path)

    # Perform Speech-to-Text Recognition
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(temp_audio_path) as source:
            recorded_speech = recognizer.record(source)
            text_transcription = recognizer.recognize_google(recorded_speech)

        st.subheader("2. Recognized Speech (Speech-to-Text)")
        st.info(text_transcription)

        # Process text via Deep Learning Model
        intent, bot_response = predict_intent(text_transcription)

        st.subheader("3. Deep Learning Intent Classification")
        st.success(f"**Predicted Intent:** {intent.capitalize()}")

        st.subheader("4. Chatbot Response")
        st.write(bot_response)

        # Generate Text-To-Speech Output
        tts = gTTS(text=bot_response, lang='en')
        out_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(out_audio_path)

        st.subheader("5. Audio Response (Text-to-Speech)")
        st.audio(out_audio_path, autoplay=True)

    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try speaking clearer.")
    except sr.RequestError as e:
        st.error(f"Speech Recognition service error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
