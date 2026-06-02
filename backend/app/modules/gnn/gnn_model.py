"""
gnn_model.py — STING DSS GNN Modeli
--------------------------------------
Orijinal WP-3 kodundan uyarlandı.
GCNConv + Neural ODE regresör.
WBC ve ANC tahminleri için genişletildi.

Bağımlılıklar (opsiyonel):
  pip install torch torch-geometric torchdiffeq
Yoksa numpy tabanlı fallback kullanılır.
"""

from __future__ import annotations
import numpy as np

# ── PyTorch / torch-geometric opsiyonel ────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    import torchdiffeq
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Neural ODE fonksiyonu  (orijinal ODEFunc.py'den)
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class ODEFunction(nn.Module):
        """dx/dt = relu(Wx) — GNN gizli durumunun zamansal evrimi."""

        def __init__(self, hidden_channels: int):
            super().__init__()
            self.fc = nn.Linear(hidden_channels, hidden_channels)

        def forward(self, t, x):
            return torch.relu(self.fc(x))


    class GNNRegressor(nn.Module):
        """
        Genişletilebilir GNN Regresör:
          - Değişken sayıda GCNConv katmanı (1–4)
          - Opsiyonel Neural ODE
          - Dropout
          - Çıkış: [WBC, ANC] veya ileride daha fazla biyobelirteç
        """

        def __init__(
            self,
            in_channels:     int,
            hidden_channels: int   = 32,
            out_channels:    int   = 2,
            n_conv_layers:   int   = 2,
            use_ode:         bool  = True,
            dropout:         float = 0.0,
        ):
            super().__init__()
            torch.manual_seed(1234567)

            # GCN katmanları — değişken sayı
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(n_conv_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

            self.use_ode = use_ode
            if use_ode:
                self.ode = ODEFunction(hidden_channels)

            self.dropout = nn.Dropout(dropout)
            self.linear  = nn.Linear(hidden_channels, out_channels)

            # Mimari bilgisi — kaydetmek için
            self.arch_info = {
                "in_channels":     in_channels,
                "hidden_channels": hidden_channels,
                "out_channels":    out_channels,
                "n_conv_layers":   n_conv_layers,
                "use_ode":         use_ode,
                "dropout":         dropout,
            }

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            for i, conv in enumerate(self.convs):
                x = torch.relu(conv(x, edge_index))
                if i < len(self.convs) - 1:
                    x = self.dropout(x)

            if self.use_ode:
                t_span = torch.tensor([0.0, 1.0], dtype=torch.float32)
                x = torchdiffeq.odeint(self.ode, x, t_span)[-1]

            return self.linear(x)
