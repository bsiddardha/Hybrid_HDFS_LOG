import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0, max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LogTransformer(nn.Module):

    def __init__(
        self,
        vocab_size=31,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=2,
        max_len=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

        self.position = PositionalEncoding(
            d_model,
            max_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, input_ids, attention_mask):

        x = self.embedding(input_ids)

        x = self.position(x)

        padding_mask = (attention_mask == 0)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (
            (x * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

        pooled = self.dropout(pooled)

        return self.classifier(pooled)