import torch
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
class SmallXVector(nn.Module):
    def __init__(self, target_dim=128, device="cuda"):
        super().__init__()
        # 載入 SpeechBrain 官方預訓練的 x-vector
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": device}
        )
        # 預設輸出是 512 維
        self.proj = nn.Linear(512, target_dim)

    def forward(self, wavs, wav_lens=None):
        """
        wavs: (batch, time) waveform tensor
        wav_lens: (batch,) 長度比例 (選用)
        """
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(wavs, wav_lens)  # (batch, 1, 512)

        embeddings = embeddings.squeeze(1)  # → (batch, 512)
        return self.proj(embeddings)  

# ---- 測試 ----
if __name__ == "__main__":
    # 模擬 batch audio (batch=4, 長度=16000)
    wavs = torch.randn(4, 16000)

    model = SmallXVector(target_dim=128, device="cpu")
    out = model(wavs)

    print("輸出 shape:", out.shape)  # (4, 128)