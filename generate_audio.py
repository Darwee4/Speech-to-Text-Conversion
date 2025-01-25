import soundfile as sf
import numpy as np

# Create a simple sine wave
sample_rate = 16000
duration = 3.0  # seconds
frequency = 440.0  # Hz
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

# Save as WAV file
sf.write('example.wav', audio_data, sample_rate)
print("Generated example.wav successfully")
