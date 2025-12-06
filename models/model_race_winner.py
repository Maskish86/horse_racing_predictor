import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    acts = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh()
    }
    return acts[name]


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, act, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(dim, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.act = act
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, dim) if hidden_dim != dim else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.proj(out)
        return residual + out


def build_mlp(in_dim, cfg):
    act = get_activation(cfg.get("act", "relu"))
    hidden_dim = cfg.get("hidden_dim", 128)
    n_layers = cfg.get("n_layers", 4)
    dropout = cfg.get("dropout", 0.1)

    layers = []
    dim = in_dim
    for _ in range(n_layers):
        layers.append(ResidualBlock(dim, hidden_dim, act, dropout))
    mlp = nn.Sequential(*layers)
    return mlp


class CrossHorseAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.get("hidden_dim", 128)
        n_heads = cfg.get("n_heads", 4)
        n_layers = cfg.get("n_layers", 1)
        dropout = cfg.get("dropout", 0.1)
        activation = cfg.get("activation", "gelu")
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, batch_first=True,
            dim_feedforward=hidden * 2, dropout=dropout, activation=activation)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, mask=None):
        attn_mask = ~mask if mask is not None else None
        return self.encoder(x, src_key_padding_mask=attn_mask)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, race_dim, hidden_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim + race_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, horse_ctx, race_emb, mask=None):
        race_expanded = race_emb.unsqueeze(1).expand(-1, horse_ctx.size(1), -1)
        x = torch.cat([horse_ctx, race_expanded], dim=-1)  
        attn_scores = self.proj(x).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(horse_ctx * attn_weights.unsqueeze(-1), dim=1)
        return pooled, attn_weights



class RaceWinnerPredictor(nn.Module):
    def __init__(self, horse_dim, race_dim,
                 horse_cfg, race_cfg, attn_cfg, head_cfg):
        super().__init__()

        self.horse_encoder = build_mlp(horse_dim, horse_cfg)
        self.race_encoder = build_mlp(race_dim, race_cfg)
        self.cross_attention = CrossHorseAttention(attn_cfg)
        self.attn_pool = AttentionPooling(attn_cfg.get("hidden", 128), race_cfg.get("hidden_dim", 128))

        input_to_head = attn_cfg["hidden_dim"] + race_cfg["hidden_dim"]
        self.win_head = build_mlp(input_to_head, head_cfg)

    def forward(self, horse_feats, race_feats, mask=None):
        horse_emb = self.horse_encoder(horse_feats)
        race_emb = self.race_encoder(race_feats)
        horse_ctx = self.cross_attention(horse_emb, mask)
        pooled_horses, attn_weights = self.attn_pool(horse_ctx, race_emb, mask)
        combined = torch.cat([pooled_horses, race_emb], dim=-1)
        logits = self.win_head(combined).squeeze(-1)
        return logits, attn_weights


