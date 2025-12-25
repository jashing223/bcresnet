# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import torch.nn.functional as F
from torch import nn

from subspectralnorm import SubSpectralNorm

# [移除] FiLMLayer 類別已刪除，因為不再使用 FiLM 機制

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()

        def get_padding(kernel_size, use_dilation):
            rate = 1  # dilation rate
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx

        # padding and dilation rate
        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        # convbnrelu block
        layers = []
        layers.append(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)
        )
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        # 2D part (f2)
        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        # 1D part (f1)
        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        # 2D part
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        # 1D part
        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNets(nn.Module):
    def __init__(self, base_c, num_classes=12, embedding_dim=192):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.n = [2, 2, 4, 4]  # identical modules repeated n times
        self.c = [
            base_c * 2,  # 1 * 8 * 2 = 16
            base_c,  # 1 * 8 = 8
            int(base_c * 1.5),  # int(1 * 8 * 1.5) = 12
            base_c * 2,  # 16
            int(base_c * 2.5),  # int(1 * 8 * 2.5) = 20
            base_c * 4,  # 1 * 8 * 4 = 32
        ]  # num channels
        self.s = [1, 2]  # stage using stride
        self._build_network()

    def _build_network(self):
        # Head: (Conv-BN-ReLU)
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        # Body: BC-ResBlocks
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        # [修改] Speech Branch (Late Fusion MLP)
        # 輸入維度: Audio Feature (self.c[-2]) + Speaker Embedding (embedding_dim)
        fusion_dim = self.c[-2] + self.embedding_dim
        
        self.classifier1 = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 3) # Same, Diff, Silence
        )

        # Keyword branch (保持不變)
        self.classifier2 = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], 1, 1),
        )

        nn.init.constant_(self.classifier2[-1].bias, -2.0)

        # Keyword classification (保持不變)
        self.classifier3 = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes-2, 1),
        )

    def encode(self, x):
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)

        return x
    
    def speech_branch(self, x, speaker_embedding):
        """Modified Speaker classification branch with Late Fusion.
        
        Args:
            x: Encoded audio features [batch, channels, H, W]
            speaker_embedding: Speaker embedding [batch, embedding_dim]
        
        Returns:
            Speaker classification logits
        """
        # 1. 將 Audio Features 轉為向量 (Global Average Pooling)
        # x: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        audio_vec = F.adaptive_avg_pool2d(x, (1, 1)).view(x.shape[0], -1)
        
        # 2. 處理 Speaker Embedding
        speaker_embedding = speaker_embedding.view(speaker_embedding.shape[0], -1) # 確保是 [B, 192]
        
        # 3. 拼接 (Concatenation)
        combined_vec = torch.cat([audio_vec, speaker_embedding], dim=1)
        
        # 4. MLP 分類
        logits = self.classifier1(combined_vec)
        
        return logits
    
    def keyword_branch(self, x):
        x = self.classifier2(x)
        x = x.view(-1, x.shape[1])

        return x
    
    def keyword_classification(self, x):
        x = self.classifier3(x)
        x = x.view(-1, x.shape[1])

        return x
    
    def forward(self, x, speaker_embedding):
        # 1. Normalize Speaker Embedding (標準操作)
        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)

        # 2. [訓練策略] Embedding Dropout 
        # 建議保留這個！防止模型過度依賴 Embedding，強迫它看 Audio
        if self.training:
            mask = torch.bernoulli(torch.full((speaker_embedding.shape[0], 1), 0.8)).to(speaker_embedding.device)
            speaker_embedding = speaker_embedding * mask
            
        # 3. 提取聲學特徵
        encoded = self.encode(x)
        
        # 4. Speech Branch (Late Fusion + 恢復梯度)
        # [關鍵修正] 移除 .detach()
        # 這樣 Speech Loss 的梯度可以回傳給 Encoder，讓 Encoder 學會編碼少量的語者資訊
        # 但因為是 Late Fusion，不會像 FiLM 那樣扭曲整張 Feature Map
        speaker_logits = self.speech_branch(encoded, speaker_embedding)
        
        # 5. Keyword Branches
        keyword_logits = self.keyword_branch(encoded)
        keyword_class_logits = self.keyword_classification(encoded)
        
        return speaker_logits, keyword_logits, keyword_class_logits

    def inference(self, x, speaker_embedding, speech_threshold=0.1, keyword_threshold=0.5):
        with torch.no_grad():
            # Define probabilities
            P_non_speech = P_non_keyword = torch.zeros(1, 1, device=x.device)
            P_keyword_id = torch.zeros(1, 10, device=x.device)

            # Extract embeddings
            encoded = self.encode(x)
            
            # [重要] Inference 時也要記得 Normalize Embedding (保持與 Training 一致)
            speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)

            # Get speaker classification
            speaker_logits = self.speech_branch(encoded, speaker_embedding)
            speaker_probs = F.softmax(speaker_logits, dim=1)      
            
            # [修正] 啟用 speech_threshold
            # 原本邏輯: if torch.argmax(speaker_probs.squeeze(0)).item() != 2:
            # 修正邏輯: 檢查 Class 2 (Same Speaker) 的機率是否小於閾值
            
            # speaker_probs shape: [1, 3] -> 取出 [0, 2] 即 Target Speaker 的機率
            target_speaker_prob = speaker_probs[0, 2]
            
            if target_speaker_prob < speech_threshold:  # 如果信心度不足，視為非目標
                P_non_speech = torch.ones(1, 1, device=x.device)
                P = torch.cat([P_non_speech, P_non_keyword, P_keyword_id], dim=1)
                return P
            
            # Step 2: Keyword vs. Non-keyword classification
            P_keyword = torch.sigmoid(self.keyword_branch(encoded))
            if P_keyword.squeeze(0) < keyword_threshold:
                P_non_keyword = torch.ones(1, 1, device=x.device)
                P = torch.cat([P_non_speech, P_non_keyword, P_keyword_id], dim=1)
                return P
                
            # Step 3: Keyword classification
            P_keyword_id = self.keyword_classification(encoded).softmax(dim=1)
            P = torch.cat([P_non_speech, P_non_keyword, P_keyword_id], dim=1)
            return P