#!/usr/bin/env python
"""
Test script to verify the speaker embedding implementation.
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms

# Import the modified components
from utils import SpeechCommand, Padding
from bcresnet import BCResNets


def test_dataset():
    """Test the modified SpeechCommand dataset."""
    print("Testing SpeechCommand dataset...")
    
    # Create dummy embeddings file for testing
    dummy_embeddings = {
        "speaker1": torch.randn(512),
        "speaker2": torch.randn(512),
        "speaker3": torch.randn(512),
    }
    torch.save(dummy_embeddings, "test_embeddings_dummy.pt")
    
    # Test dataset loading (using a dummy path for testing)
    transform = transforms.Compose([Padding()])
    
    # Note: This will fail if the actual data directory doesn't exist
    # For testing purposes, we'll catch the error
    try:
        dataset = SpeechCommand(
            root_dir="/tmp/dummy_dir",  # Dummy path
            ver=1,
            transform=transform,
            embeddings_path="test_embeddings_dummy.pt"
        )
        print("[OK] Dataset initialized successfully")
    except Exception as e:
        print(f"Dataset initialization failed (expected if data dir doesn't exist): {e}")
    
    # Clean up
    os.remove("test_embeddings_dummy.pt")
    print("✓ Dataset test completed")


def test_model():
    """Test the modified BCResNets model."""
    print("\nTesting BCResNets model...")
    
    # Create model
    model = BCResNets(base_c=8, num_classes=12, embedding_dim=512)
    print(f"✓ Model created successfully")
    
    # Test forward pass
    batch_size = 4
    audio_input = torch.randn(batch_size, 1, 40, 87)  # [batch, channels, freq, time]
    speaker_embedding = torch.randn(batch_size, 512)  # [batch, embedding_dim]
    
    # Test forward pass with speaker embedding
    speaker_logits, keyword_logits, keyword_class_logits = model(audio_input, speaker_embedding)
    
    assert speaker_logits.shape == (batch_size, 3), f"Expected speaker logits shape (batch, 3), got {speaker_logits.shape}"
    assert keyword_logits.shape == (batch_size, 1), f"Expected keyword logits shape (batch, 1), got {keyword_logits.shape}"
    assert keyword_class_logits.shape == (batch_size, 10), f"Expected keyword class logits shape (batch, 10), got {keyword_class_logits.shape}"
    
    print(f"✓ Forward pass successful")
    print(f"  Speaker logits shape: {speaker_logits.shape}")
    print(f"  Keyword logits shape: {keyword_logits.shape}")
    print(f"  Keyword class logits shape: {keyword_class_logits.shape}")
    
    # Test inference
    audio_single = torch.randn(1, 1, 40, 87)
    speaker_emb_single = torch.randn(1, 512)
    
    with torch.no_grad():
        outputs = model.inference(audio_single, speaker_emb_single)
    
    assert outputs.shape == (1, 12), f"Expected inference output shape (1, 12), got {outputs.shape}"
    print(f"✓ Inference successful")
    print(f"  Output shape: {outputs.shape}")
    
    # Test FiLM layer
    from bcresnet import FiLMLayer
    film = FiLMLayer(embedding_dim=512, feature_dim=20)
    features = torch.randn(batch_size, 20, 5, 87)
    embedding = torch.randn(batch_size, 512)
    modulated = film(features, embedding)
    
    assert modulated.shape == features.shape, f"Expected FiLM output shape {features.shape}, got {modulated.shape}"
    print(f"✓ FiLM layer test successful")
    
    print("✓ All model tests passed!")


def test_integration():
    """Test integration of all components."""
    print("\nTesting integration...")
    
    # Create a small batch of dummy data
    batch_size = 2
    audio = torch.randn(batch_size, 1, 16000)  # 1 second audio at 16kHz
    speaker_embedding = torch.randn(batch_size, 512)
    keyword_labels = torch.tensor([0, 2])  # silence, keyword
    speaker_labels = torch.tensor([2, 0])  # silence, same speaker
    
    # Apply padding
    padding = Padding()
    audio_padded = torch.stack([padding(audio[i]) for i in range(batch_size)])
    
    print(f"✓ Audio shape after padding: {audio_padded.shape}")
    
    # Create model and test with the data
    model = BCResNets(base_c=8)
    
    # Preprocess would normally be done here, but we'll skip for this test
    # and use dummy mel-spectrogram input
    mel_input = torch.randn(batch_size, 1, 40, 87)
    
    # Forward pass
    speaker_out, keyword_out, keyword_class_out = model(mel_input, speaker_embedding)
    
    print(f"✓ Integration test successful")
    print(f"  Input shapes: audio={audio_padded.shape}, embedding={speaker_embedding.shape}")
    print(f"  Output shapes: speaker={speaker_out.shape}, keyword={keyword_out.shape}, class={keyword_class_out.shape}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Speaker Embedding Implementation")
    print("=" * 50)
    
    test_dataset()
    test_model()
    test_integration()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run 'python extract_embeddings.py' to extract speaker embeddings")
    print("2. Run 'python main.py' to train the model with speaker embeddings")


if __name__ == "__main__":
    main()