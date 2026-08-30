"""
AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
Reference: Jung et al., AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks (Interspeech 2022)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SincConv(nn.Module):
    """
    Sinc-based convolution layer directly processing raw audio waveforms
    """
    def __init__(self, out_channels=70, kernel_size=128, in_channels=1, sample_rate=16000):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1

        # Initialize filterbanks between 30Hz and Nyquist
        hz = torch.linspace(30, sample_rate / 2 - 100, out_channels + 1)
        mel = 2595 * torch.log10(1 + hz / 700)
        melf = torch.linspace(mel[0], mel[-1], out_channels + 1)
        hz = 700 * (10 ** (melf / 2595) - 1)

        self.f_min = nn.Parameter(hz[:-1].unsqueeze(1))
        self.f_max = nn.Parameter(hz[1:].unsqueeze(1))

        t_right = torch.linspace(1, (self.kernel_size - 1) / 2, steps=int((self.kernel_size - 1) / 2)) / sample_rate
        self.register_buffer('t_right', t_right)

        # Hamming window
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.linspace(0, self.kernel_size - 1, steps=self.kernel_size) / self.kernel_size)
        self.register_buffer('window', window)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        f_min = torch.abs(self.f_min)
        f_max = f_min + torch.abs(self.f_max - f_min)

        t_right = self.t_right.unsqueeze(0)
        f_min_t = 2 * math.pi * f_min * t_right
        f_max_t = 2 * math.pi * f_max * t_right

        band_pass_right = (torch.sin(f_max_t) - torch.sin(f_min_t)) / (2 * math.pi * t_right * self.sample_rate)
        band_pass_left = torch.flip(band_pass_right, dims=[-1])
        band_pass_center = 2 * (f_max - f_min) / self.sample_rate

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=-1)
        band_pass = band_pass * self.window.unsqueeze(0)
        filters = band_pass.unsqueeze(1)

        return F.conv1d(x, filters, stride=1, padding=(self.kernel_size - 1) // 2)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.3)
        self.pool = nn.MaxPool1d(3)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        out = out + res
        out = self.pool(out)
        return out

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for spectro-temporal artifact modeling
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h: [Batch, Nodes, Features]
        Wh = self.W(h)
        B, N, F_dim = Wh.size()
        
        Wh1 = Wh.repeat_interleave(N, dim=1)
        Wh2 = Wh.repeat(1, N, 1)
        all_combinations = torch.cat([Wh1, Wh2], dim=-1)
        
        e = self.leakyrelu(self.a(all_combinations).view(B, N, N))
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        h_prime = torch.bmm(attention, Wh)
        return F.elu(h_prime)

class AASIST(nn.Module):
    """
    Standalone AASIST Architecture for Raw Waveform Audio Anti-Spoofing
    """
    def __init__(self, num_classes=2, sample_rate=16000):
        super(AASIST, self).__init__()
        self.sinc_conv = SincConv(out_channels=70, kernel_size=128, sample_rate=sample_rate)
        self.bn0 = nn.BatchNorm1d(70)
        self.act = nn.LeakyReLU(0.3)

        self.block1 = ResidualBlock(70, 32)
        self.block2 = ResidualBlock(32, 64)
        self.block3 = ResidualBlock(64, 64)

        self.gat = GraphAttentionLayer(in_features=64, out_features=64)
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [Batch, Samples] or [Batch, 1, Samples]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        feat = self.act(self.bn0(self.sinc_conv(x)))
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)

        # Permute for graph attention: [Batch, Time, Channels]
        h = feat.permute(0, 2, 1)
        h_gat = self.gat(h)

        # Global Pooling (Mean + Max)
        mean_pool = torch.mean(h_gat, dim=1)
        max_pool, _ = torch.max(h_gat, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)

        logits = self.fc(pooled)
        return logits
