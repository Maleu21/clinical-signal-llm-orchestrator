from __future__ import annotations
import torch
import torch.nn as nn


class ECGConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, 64, 1)
        )
        self.head = nn.Linear(64, 2)  # binary classification as 2 logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).squeeze(-1)   # (B, 64)
        return self.head(x)           # (B, 2)