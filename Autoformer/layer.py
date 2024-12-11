import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional Encoding for time-series data."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, : x.size(1), :]


class DecompositionLayer(nn.Module):
    """Decomposes time series into trend and seasonal components."""
    def __init__(self, kernel_size):
        super(DecompositionLayer, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim] or [batch_size, seq_len, feature_dim, 1]
        if len(x.shape) == 4:
            # Handle [batch_size, seq_len, feature_dim, 1]
            x = x.squeeze(-1)  # [batch_size, seq_len, feature_dim]

        # Apply moving average on dimension representing time
        # For AvgPool1d: input should be [batch, channels, sequence]
        # Here: batch = batch_size, channels = feature_dim, sequence = seq_len
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_len, feature_dim]
        seasonal = x - trend
        return seasonal, trend


class EncoderLayer(nn.Module):
    """Single Encoder Layer."""
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        attn_out, _ = self.attn(x, x, x)  # self-attention
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x  # [seq_len, batch_size, d_model]


class Encoder(nn.Module):
    """Autoformer Encoder."""
    def __init__(self, d_model, n_heads, ff_dim, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        for layer in self.layers:
            x = layer(x)
        return x  # [seq_len, batch_size, d_model]


class DecoderLayer(nn.Module):
    """Single Decoder Layer."""
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        # x, enc_output: [seq_len, batch_size, d_model]
        # Self-attention
        self_attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention
        cross_attn_out, _ = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x  # [seq_len, batch_size, d_model]


class Decoder(nn.Module):
    """Autoformer Decoder."""
    def __init__(self, d_model, n_heads, ff_dim, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_output):
        # x: [seq_len, batch_size, d_model]
        # enc_output: [seq_len, batch_size, d_model]
        for layer in self.layers:
            x = layer(x, enc_output)
        return x  # [seq_len, batch_size, d_model]


class Autoformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, ff_dim, num_layers, kernel_size, target_len, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.decomposition = DecompositionLayer(kernel_size)
        self.encoder = Encoder(d_model, n_heads, ff_dim, num_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, ff_dim, num_layers, dropout)
        self.output_projection = nn.Linear(d_model, 1)

    def prepare_decoder_input(self, target):
        shifted_target = torch.zeros_like(target)
        shifted_target[:, 1:] = target[:, :-1]
        return shifted_target

    def forward(self, x, target):
        
        seasonal, trend = self.decomposition(x)  
        enc_input = self.input_projection(seasonal)

        enc_input = self.positional_encoding(enc_input)  
        enc_input = enc_input.permute(1, 0, 2)

        enc_output = self.encoder(enc_input) 

        dec_input = self.prepare_decoder_input(target)

        # dec_input = self.input_projection(dec_input)
        dec_input = self.positional_encoding(dec_input)
        dec_input = dec_input.permute(1, 0, 2)

        dec_output = self.decoder(dec_input, enc_output)
        dec_output = dec_output.permute(1, 0, 2)

        output = self.output_projection(dec_output).squeeze(-1) 
        return output
