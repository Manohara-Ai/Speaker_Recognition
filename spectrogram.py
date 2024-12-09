import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import os

# Configuration
samples = 5
duration = 5  # Recording duration in seconds
sample_rate = 16000  # Sampling rate
channels = 1

def generate_spectrogram(user_name, output_dir="buffer"):
    # Create the user directories
    user_audio_dir = os.path.join(output_dir, user_name, "audio")
    user_spectrogram_dir = os.path.join("datasets", user_name)
    os.makedirs(user_audio_dir, exist_ok=True)
    os.makedirs(user_spectrogram_dir, exist_ok=True)

    # Record audio for the user
    for i in range(samples):
        # Record audio
        print(f"Recording {i+1} for {duration} seconds... Speak now!")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()

        audio_filename = os.path.join(user_audio_dir, f"recording_{i+1}.wav")
        write(audio_filename, sample_rate, audio)
        print(f"Audio saved to {audio_filename}")

        # Generate the spectrogram for each audio
        y, sr = librosa.load(audio_filename, sr=sample_rate)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 6))
        librosa.display.specshow(S_db, sr=sr, hop_length=512, cmap='viridis')
        plt.axis('off')

        spectrogram_filename = os.path.join(user_spectrogram_dir, f"spectrogram_{i+1}.png")
        plt.savefig(spectrogram_filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Spectrogram saved to {spectrogram_filename}")

generate_spectrogram("testing")
