#!/usr/bin/env python
"""
Extract speaker embeddings using SpeechBrain's pre-trained x-vector model.
For each unique speaker ID (first 8 characters of filename), randomly select
one audio file and extract its embedding.
"""

import os
import random
from glob import glob
from collections import defaultdict
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


def get_speaker_id(filename):
    """Extract speaker ID from filename (first 8 characters)."""
    basename = os.path.basename(filename)
    # Remove file extension
    basename = basename.replace('.wav', '')
    # Get first 8 characters as speaker ID
    if len(basename) >= 8:
        return basename[:8]
    return basename


def group_files_by_speaker(audio_files):
    """Group audio files by speaker ID."""
    speaker_groups = defaultdict(list)
    for filepath in audio_files:
        speaker_id = get_speaker_id(filepath)
        speaker_groups[speaker_id].append(filepath)
    return speaker_groups


def extract_embeddings(data_dir, output_path):
    """
    Extract speaker embeddings for a dataset directory.
    
    Args:
        data_dir: Path to the dataset directory (train/valid/test)
        output_path: Path to save the embeddings .pt file
    """
    print(f"Processing {data_dir}...")
    
    # Load pre-trained x-vector model
    print("Loading SpeechBrain x-vector model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
    
    # Get all audio files
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {data_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Group files by speaker
    speaker_groups = group_files_by_speaker(audio_files)
    print(f"Found {len(speaker_groups)} unique speakers")
    
    # Extract embeddings
    embeddings_dict = {}
    
    for speaker_id, file_list in tqdm(speaker_groups.items(), desc="Extracting embeddings"):
        # Randomly select one file for this speaker
        selected_file = random.choice(file_list)
        
        try:
            # Load audio
            signal, fs = torchaudio.load(selected_file)
            
            # Resample if necessary (x-vector model expects 16kHz)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)
            
            # Ensure single channel
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            
            # Extract embedding
            with torch.no_grad():
                embeddings = classifier.encode_batch(signal)
                # Get the embedding (squeeze batch dimension)
                embedding = embeddings.squeeze(0).cpu()
            
            embeddings_dict[speaker_id] = embedding
            
        except Exception as e:
            print(f"Error processing {selected_file}: {e}")
            # Use zero embedding as fallback
            embeddings_dict[speaker_id] = torch.zeros(512)  # x-vector dimension is 512
    
    # Save embeddings
    torch.save(embeddings_dict, output_path)
    print(f"Saved embeddings to {output_path}")
    
    return embeddings_dict


def prepare_embedding(base_dir: str):
    """Main function to extract embeddings for all datasets."""
    
    # Check if v2 exists, use it if available
    base_dir_v2 = base_dir.replace("v0.01", "v0.02")
    if os.path.exists(base_dir_v2):
        base_dir = base_dir_v2
        print("Using GSC v2")
    else:
        print("Using GSC v1")
    
    # Process train, valid, and test sets
    datasets = [
        (f"{base_dir}/train_12class", "train_embeddings.pt"),
        (f"{base_dir}/valid_12class", "valid_embeddings.pt"),
    ]
    
    # For test set, use the test_set directory
    test_dir = base_dir.replace("commands", "commands_test_set")
    if os.path.exists(test_dir):
        datasets.append((test_dir, "test_embeddings.pt"))
    
    for data_dir, output_file in datasets:
        if os.path.exists(data_dir):
            output_path = os.path.join(os.path.dirname(__file__), output_file)
            extract_embeddings(data_dir, output_path)
        else:
            print(f"Directory not found: {data_dir}")
    
    print("Embedding extraction completed!")


if __name__ == "__main__":
    prepare_embedding()