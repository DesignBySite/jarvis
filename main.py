import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3
from transformers import pipeline

# Load a conversational model
generator = pipeline('text-generation', model='distilgpt2')


# Initialize Whisper and pyttsx3
model = whisper.load_model("base", device="cpu")  # Ensure it runs on CPU with FP32
engine = pyttsx3.init()

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to record audio and save it as a .wav file
def record_audio(file_name="input.wav", duration=3, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(file_name, fs, recording)
    print("Recording complete!")

# Function to use Whisper for speech-to-text
def transcribe_audio(file_name="input.wav"):
    result = model.transcribe(file_name)
    return result["text"]

# Updated greeting function using Whisper and pyttsx3
def chat(name):
    speak(f"Hello, {name}. How are you?")
    record_audio()  # Record the user's response
    text = transcribe_audio()  # Transcribe the recorded response
    conversation = generator(text, max_length=20, truncation=True)
    speak(conversation[0]['generated_text'])


# Start the greeting function
while True:
  chat("Kevin")
