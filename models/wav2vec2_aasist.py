"""
Wav2Vec2-XLS-R + AASIST / GAT Audio Anti-Spoofing Architecture
Combines self-supervised multilingual acoustic representations with Graph Attention Networks
for state-of-the-art accent robustness and synthetic voice artifact detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

class GraphAttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super(GraphAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [Batch, Time, Features]
        B, T, C = x.size()
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, self.out_dim)
        out = self.out_proj(context)
        return self.layer_norm(x + out)

class Wav2Vec2AASIST(nn.Module):
    """
    Fine-tuneable Wav2Vec2-XLS-R Front-End + AASIST/GAT Back-End
    """
    def __init__(
        self,
        ssl_model_name: str = "facebook/wav2vec2-xls-r-300m",
        fallback_model_name: str = "facebook/wav2vec2-base",
        freeze_ssl_layers: int = 18,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super(Wav2Vec2AASIST, self).__init__()
        
        # Load Pretrained Wav2Vec2 backbone
        try:
            print(f"Loading SSL Front-End: {ssl_model_name}")
            self.ssl_model = Wav2Vec2Model.from_pretrained(ssl_model_name)
        except Exception as e:
            print(f"Warning: Failed to load {ssl_model_name} ({e}). Falling back to {fallback_model_name}")
            self.ssl_model = Wav2Vec2Model.from_pretrained(fallback_model_name)

        ssl_hidden_size = self.ssl_model.config.hidden_size

        # Freeze early layers for efficient Colab fine-tuning
        self.ssl_model.feature_extractor._freeze_parameters()
        if hasattr(self.ssl_model, "encoder") and hasattr(self.ssl_model.encoder, "layers"):
            total_layers = len(self.ssl_model.encoder.layers)
            freeze_count = min(freeze_ssl_layers, total_layers - 2)
            print(f"Freezing bottom {freeze_count}/{total_layers} transformer encoder layers")
            for i in range(freeze_count):
                for param in self.ssl_model.encoder.layers[i].parameters():
                    param.requires_grad = False

        # Anti-Spoofing Graph Attention Backend
        self.proj = nn.Linear(ssl_hidden_size, hidden_dim)
        self.gat1 = GraphAttentionBlock(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
        self.gat2 = GraphAttentionBlock(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [Batch, Samples] (raw audio waveform @ 16kHz)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        # Extract self-supervised speech representations
        ssl_outputs = self.ssl_model(x)
        features = ssl_outputs.last_hidden_state  # [Batch, Frames, Hidden]

        # Project and apply Graph Attention
        h = self.proj(features)
        h = self.gat1(h)
        h = self.gat2(h)

        # Multi-pooling (Mean + Max across temporal frames)
        mean_pool = torch.mean(h, dim=1)
        max_pool, _ = torch.max(h, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)

        # Logits [Batch, num_classes] (0: Bonafide, 1: Spoof)
        logits = self.classifier(pooled)
        return logits

class Wav2Vec2Classifier(nn.Module):
    """
    Lightweight Wav2Vec2 Classifier with Linear Head (fast baseline)
    """
    def __init__(self, ssl_model_name="facebook/wav2vec2-base", num_classes=2, dropout=0.2):
        super(Wav2Vec2Classifier, self).__init__()
        self.ssl_model = Wav2Vec2Model.from_pretrained(ssl_model_name)
        self.ssl_model.feature_extractor._freeze_parameters()
        
        ssl_hidden = self.ssl_model.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(ssl_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        out = self.ssl_model(x).last_hidden_state
        pooled = torch.mean(out, dim=1)
        return self.head(pooled)
