import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3

# Initialize Whisper and pyttsx3
model = whisper.load_model("base")
engine = pyttsx3.init()

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to record audio and save it as a .wav file
def record_audio(file_name="input.wav", duration=5, fs=44100):
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
def greeting(name):
    speak(f"Hello, {name}. How are you?")
    record_audio()  # Record the user's response
    text = transcribe_audio()  # Transcribe the recorded response
    print(f"You said: {text}")

    # Respond based on the text
    if "good" in text.lower():
        speak("That's great to hear!")
    else:
        speak("It's going to be okay.")

# Start the greeting function
greeting("Kevin")
