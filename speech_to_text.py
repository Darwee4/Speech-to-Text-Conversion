import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
import soundfile as sf

def load_librispeech_dataset():
    """Load and preprocess LibriSpeech dataset"""
    # Load dataset with audio resampling
    dataset = load_dataset("librispeech_asr", "clean", split="train.100", 
                          trust_remote_code=True)
    
    # Resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Preprocess function to extract input values
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = processor(audio_arrays, sampling_rate=16000, 
                         padding=True, return_tensors="pt")
        return inputs
    
    return dataset, preprocess_function

def setup_wav2vec2_model():
    """Initialize Wav2Vec 2.0 model and processor"""
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

def train_model(dataset, preprocess_function, processor, model):
    """Configure and train the model"""
    # Prepare dataset
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        fp16=True
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    return trainer

def transcribe_audio(file_path, processor, model):
    """Transcribe audio file using the trained model"""
    # Load audio file
    audio_input, _ = sf.read(file_path)
    
    # Preprocess audio
    inputs = processor(audio_input, sampling_rate=16000, 
                      return_tensors="pt", padding=True)
    
    # Get logits from model
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # Decode predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription

if __name__ == "__main__":
    # Initialize components
    processor, model = setup_wav2vec2_model()
    dataset, preprocess_function = load_librispeech_dataset()
    
    # Train model
    trainer = train_model(dataset, preprocess_function, processor, model)
    trainer.train()
    
    # Example usage of transcription
    transcription = transcribe_audio("example.wav", processor, model)
    print("Transcription:", transcription)
