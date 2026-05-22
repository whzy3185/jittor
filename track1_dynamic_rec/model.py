from __future__ import annotations

import math

import jittor as jt
from jittor import nn


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int, shift: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.shift = float(shift)
        self.scale = float(scale) if float(scale) > 0 else 1.0
        half = max(1, dim // 2)
        freqs = [1.0 / (10000 ** (i / max(1, half - 1))) for i in range(half)]
        self.freqs = jt.array(freqs).float32()

    def execute(self, t: jt.Var) -> jt.Var:
        x = ((t.view(-1, 1).float32() - self.shift) / self.scale) * self.freqs.view(1, -1)
        enc = jt.concat([jt.sin(x), jt.cos(x)], dim=1)
        if enc.shape[1] > self.dim:
            enc = enc[:, : self.dim]
        elif enc.shape[1] < self.dim:
            enc = jt.concat([enc, jt.zeros((enc.shape[0], self.dim - enc.shape[1]))], dim=1)
        return enc


class GELU(nn.Module):
    def execute(self, x: jt.Var) -> jt.Var:
        return nn.gelu(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        return x + self.net(x)


class HybridTemporalScorer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        emb_dim: int = 128,
        time_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.15,
        time_shift: float = 0.0,
        time_scale: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.emb_dim = int(emb_dim)
        self.node_embedding = nn.Embedding(self.num_nodes + 2, emb_dim)
        self.time_encoder = SinusoidalTimeEncoding(time_dim, shift=time_shift, scale=time_scale)
        input_dim = emb_dim * 4 + time_dim + num_features
        layers = [nn.Linear(input_dim, hidden_dim), GELU(), nn.Dropout(dropout)]
        for _ in range(max(1, num_layers - 1)):
            layers.append(ResidualBlock(hidden_dim, dropout))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.dot_scale = 1.0 / math.sqrt(max(1, emb_dim))
        self.dot_weight = jt.Var([0.25]).float32()

    def score(self, src: jt.Var, dst: jt.Var, t: jt.Var, feat: jt.Var) -> jt.Var:
        src = src.int()
        dst = dst.int()
        se = self.node_embedding(src)
        de = self.node_embedding(dst)
        te = self.time_encoder(t)
        x = jt.concat([se, de, se * de, jt.abs(se - de), te, feat.float32()], dim=1)
        mlp_score = self.mlp(x).view(-1)
        dot_score = (se * de).sum(dim=1) * self.dot_scale
        return mlp_score + self.dot_weight * dot_score

    def execute(self, src: jt.Var, dst: jt.Var, t: jt.Var, feat: jt.Var) -> jt.Var:
        return self.score(src, dst, t, feat)

    def pairwise_loss(
        self,
        src: jt.Var,
        pos_dst: jt.Var,
        neg_dst: jt.Var,
        t: jt.Var,
        pos_feat: jt.Var,
        neg_feat: jt.Var,
    ) -> tuple[jt.Var, jt.Var, jt.Var]:
        pos_score = self.score(src, pos_dst, t, pos_feat)
        k = neg_dst.shape[1]
        src_rep = src.unsqueeze(1).repeat(1, k).reshape(-1)
        t_rep = t.unsqueeze(1).repeat(1, k).reshape(-1)
        neg_score = self.score(src_rep, neg_dst.reshape(-1), t_rep, neg_feat.reshape((-1, neg_feat.shape[-1]))).reshape((-1, k))
        loss = -jt.log(jt.sigmoid(pos_score.unsqueeze(1) - neg_score) + 1e-10).mean()
        return loss, pos_score, neg_score
