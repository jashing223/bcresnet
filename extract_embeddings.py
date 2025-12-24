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
import numpy as np
import torchaudio.compliance.kaldi as kaldi
from modelscope.models import Model
from modelscope.utils.config import Config
from modelscope.hub.snapshot_download import snapshot_download
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
    Extract speaker embeddings for a dataset directory using ERes2NetV2.
    
    Args:
        data_dir: Path to the dataset directory (train/valid/test)
        output_path: Path to save the embeddings .pt file
    """
    print(f"Processing {data_dir}...")
    
    # 手動處理設定檔以解決 KeyError: 'device'
    model_id = 'iic/speech_eres2netv2_sv_zh-cn_16k-common'
    print(f"Loading {model_id}...")
    
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    # 下載並手動修改 Config
    model_dir = snapshot_download(model_id, revision='v1.0.2')
    cfg = Config.from_file(os.path.join(model_dir, 'configuration.json'))
    
    # 強制注入模型需要的 device 資訊
    if not hasattr(cfg.model, 'device'):
        cfg.model['device'] = device_name
    
    model = Model.from_pretrained(model_dir, cfg_dict=cfg)
    model.to(device)
    model.eval()

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
            # 載入音訊
            wav, sr = torchaudio.load(selected_file)
            
            # 1. 確保採樣率為 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
            
            # 2. 強制單聲道 [Channels, Samples] -> [1, Samples]
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # 3. 確保形狀為 [N, T] (這裡是 [1, Samples])
            # 符合錯誤訊息要求：the shape of input audio to model needs to be [N, T]
            wav = wav.to(device)
            
            # 直接推理提取 192 維 Embedding
            with torch.no_grad():
                # ERes2NetV2 模型內部會自行處理 Fbank 轉換
                embedding = model(wav)
                # [關鍵修正] 進行 L2 歸一化，將 Norm 固定為 1
                # embedding = torch.nn.functional.normalize(embedding.squeeze().cpu(), p=2, dim=0)
            
            embeddings_dict[speaker_id] = embedding
            
        except Exception as e:
            print(f"Error processing {selected_file}: {e}")
            # 修正 Fallback 維度為 192
            embeddings_dict[speaker_id] = torch.zeros(192)
    
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
    prepare_embedding("/home/jashing223/datasets/GSC/speech_commands_v0.01")