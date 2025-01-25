# Speech-to-Text Conversion with Wav2Vec 2.0

## Overview
This repository contains an implementation of a speech-to-text conversion system using the Wav2Vec 2.0 transformer-based architecture. The system is trained on the LibriSpeech dataset and provides functionality for transcribing audio files.

## Features
- Pre-trained Wav2Vec 2.0 model fine-tuning
- LibriSpeech dataset preprocessing and loading
- Audio transcription functionality
- Training configuration with hyperparameter tuning
- Support for 16kHz audio input

## Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- SoundFile
- Datasets

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Train the model:
```python
python speech_to_text.py
```

2. Transcribe audio:
```python
from speech_to_text import transcribe_audio

transcription = transcribe_audio("example.wav")
print(transcription)
```

## Model Details
- Base Model: facebook/wav2vec2-base-960h
- Training Epochs: 3
- Batch Size: 8
- Learning Rate: 3e-5
- Weight Decay: 0.01

## License
MIT License
