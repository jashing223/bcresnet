# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import torch.nn.functional as F
from torch import nn

from subspectralnorm import SubSpectralNorm


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.
    
    Takes speaker embedding as input and generates gamma and beta
    for modulating the encoded features.
    """
    def __init__(self, feature_dim, embedding_dim=64 , hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        
        # Network to generate gamma and beta from speaker embedding
        self.film_generator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, feature_dim * 2)  # Generate both gamma and beta
        )
        
    def forward(self, features, embedding):
        """
        Args:
            features: Encoded audio features [batch, channels, height, width]
            embedding: Speaker embedding [batch, embedding_dim]
        
        Returns:
            Modulated features [batch, channels, height, width]
        """
        gamma_beta = self.film_generator(embedding)
        
        # Generate gamma and beta
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # Apply FiLM modulation
        modulated = gamma * features + beta
        
        return modulated


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
    def __init__(self, base_c, num_classes=12, embedding_dim=512):
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

        # FiLM layer for speaker conditioning
        self.film_layer = FiLMLayer(embedding_dim=self.embedding_dim, feature_dim=self.c[-2])

        # Speech branch (now acts as speaker classifier with 3 classes)
        self.classifier1 = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], 3, 1),  # 3 classes: same speaker, different speaker, silence
        )

        # Keyword branch
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

        # Keyword classification
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
        """Speaker classification branch with FiLM conditioning.
        
        Args:
            x: Encoded features
            speaker_embedding: Speaker embedding for FiLM modulation
        
        Returns:
            Speaker classification logits (3 classes)
        """
        speaker_embedding = speaker_embedding.squeeze(1)
        speaker_embedding = F.layer_norm(speaker_embedding)
        x = self.film_layer(x, speaker_embedding)
        x = self.classifier1(x)
        x = x.view(-1, x.shape[1])
        return x
    
    def keyword_branch(self, x):
        x = self.classifier2(x)
        x = x.view(-1, x.shape[1])

        return x
    
    def keyword_classification(self, x):
        x = self.classifier3(x)
        x = x.view(-1, x.shape[1])

        return x
    
    def forward(self, x, speaker_embedding):
        """Forward pass with optional speaker embedding.
        
        Args:
            x: Input audio features
            speaker_embedding: Optional speaker embedding for FiLM
        
        Returns:
            Tuple of (speaker_logits, keyword_logits, keyword_class_logits)
        """
        # Encode audio features
        encoded = self.encode(x)
        
        # Speaker branch with FiLM conditioning
        speaker_logits = self.speech_branch(encoded, speaker_embedding)
        
        # Keyword branches without conditioning
        keyword_logits = self.keyword_branch(encoded)
        keyword_class_logits = self.keyword_classification(encoded)
        
        return speaker_logits, keyword_logits, keyword_class_logits

    def inference(self, x, speaker_embedding, keyword_threshold=0.5):  # batch = 1
        with torch.no_grad():
            # Define probabilities
            P_non_speech = P_non_keyword = torch.zeros(1, 1, device=x.device)
            P_keyword_id = torch.zeros(1, 10, device=x.device)

            # Extract embeddings
            encoded = self.encode(x)

            # Get speaker classification (now includes speaker verification)
            # Note: We still use this for speech detection logic
            speaker_logits = self.speech_branch(encoded, speaker_embedding)
            speaker_probs = F.softmax(speaker_logits, dim=1)        
            
            if torch.argmax(speaker_probs, dim=-1) != 2:  # if non-speech
                P_non_speech = torch.ones(1, 1, device=x.device)
                P = torch.cat([P_non_speech, P_non_keyword, P_keyword_id], dim=1)
                return P
            
            # Step 2: Keyword vs. Non-keyword classification (within speech)
            P_keyword = torch.sigmoid(self.keyword_branch(encoded))  # [batch, 1] -> P(keyword)
            if P_keyword.squeeze(0) < keyword_threshold:  # if non-keyword
                P_non_keyword = torch.ones(1, 1, device=x.device)
                P = torch.cat([P_non_speech, P_non_keyword, P_keyword_id], dim=1)
                return P
                
            # Step 3: Keyword classification (only if keyword is detected)
            P_keyword_id = self.keyword_classification(encoded).softmax(dim=1)
            P = torch.cat([P_non_speech, P_non_keyword, P_keyword_id], dim=1)
            return P
