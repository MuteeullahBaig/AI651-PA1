"""A compact Autoformer (Wu et al., NeurIPS 2021) for univariate forecasting with optional covariates.

Written for this assignment, following the paper's Sections 3.1-3.2 and the structure of the official
implementation (https://github.com/thuml/Autoformer: models/Autoformer.py, layers/AutoCorrelation.py,
layers/Autoformer_EncDec.py, layers/Embed.py). Differences from the official code, all deliberate:

* Aggregation reads values from t - tau, matching R(tau) = sum_t q_t k_{t-tau} (the Task 1 convention);
  the official code rolls by -tau.
* Delays are chosen per example in training as well as inference (the official code shares them across
  the batch during training).
* The official "time mark" path (a linear embedding of calendar features) carries the external covariates
  instead, because this series has no timestamps. That path is the only per-step input that reaches the
  decoder's future positions, which is how known-future covariates enter the forecast.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MovingAverage(nn.Module):
    """Centered moving average along time with replicate padding; x [B, L, C] -> [B, L, C]."""

    def __init__(self, kernel: int):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x):
        half = self.kernel // 2
        padded = torch.cat([x[:, :1].expand(-1, half, -1), x, x[:, -1:].expand(-1, half, -1)], dim=1)
        return F.avg_pool1d(padded.transpose(1, 2), self.kernel, stride=1).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """Returns (seasonal, trend) with seasonal + trend == x."""

    def __init__(self, kernel: int):
        super().__init__()
        self.average = MovingAverage(kernel)

    def forward(self, x):
        trend = self.average(x)
        return x - trend, trend


def aggregate_delays(values, delays, weights):
    """values [B, H, E, L]; delays, weights [B, K] -> sum_j w_j * values[..., (t - tau_j) mod L]."""
    batch, heads, channels, length = values.shape
    k = delays.shape[1]
    positions = torch.arange(length, device=values.device)
    index = (positions.view(1, 1, length) - delays.view(batch, k, 1)) % length
    index = index.view(batch, k, 1, 1, length).expand(batch, k, heads, channels, length)
    shifted = torch.gather(values.unsqueeze(1).expand(batch, k, heads, channels, length), -1, index)
    return (weights.view(batch, k, 1, 1, 1) * shifted).sum(1)


class AutoCorrelation(nn.Module):
    """Period-based dependency discovery (FFT autocorrelation) and time-delay aggregation."""

    def __init__(self, d_model: int, n_heads: int, factor: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads, self.factor = n_heads, factor
        self.query, self.key, self.value = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys):
        batch, length, width = queries.shape
        source = keys.shape[1]
        shape_q = (batch, length, self.heads, width // self.heads)
        shape_k = (batch, source, self.heads, width // self.heads)
        q = self.query(queries).view(shape_q)
        k, v = self.key(keys).view(shape_k), self.value(keys).view(shape_k)
        if length > source:          # as in the official code: zero-pad or truncate keys/values to length L
            pad = torch.zeros(batch, length - source, self.heads, width // self.heads, device=q.device, dtype=q.dtype)
            k, v = torch.cat([k, pad], 1), torch.cat([v, pad], 1)
        else:
            k, v = k[:, :length], v[:, :length]
        q, k, v = (t.permute(0, 2, 3, 1) for t in (q, k, v))                 # [B, H, E, L]
        corr = torch.fft.irfft(torch.fft.rfft(q, dim=-1) * torch.conj(torch.fft.rfft(k, dim=-1)),
                               n=length, dim=-1)                               # R(tau), tau = 0..L-1
        score = corr.mean(dim=(1, 2))                                          # [B, L]
        top_k = max(1, int(self.factor * math.log(length)))
        selected, delays = torch.topk(score, top_k, dim=-1)
        weights = torch.softmax(selected, dim=-1)
        mixed = aggregate_delays(v, delays, weights)                           # [B, H, E, L]
        return self.out(self.dropout(mixed.permute(0, 3, 1, 2).reshape(batch, length, width)))


class SeasonalLayerNorm(nn.Module):
    """LayerNorm followed by removing the time-mean (the official `my_Layernorm`)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(dim=1, keepdim=True)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model, bias=False), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, kernel, factor, dropout):
        super().__init__()
        self.correlation = AutoCorrelation(d_model, n_heads, factor)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.decomp1, self.decomp2 = SeriesDecomposition(kernel), SeriesDecomposition(kernel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.decomp1(x + self.dropout(self.correlation(x, x)))
        x, _ = self.decomp2(x + self.feed_forward(x))
        return x                                               # seasonal part only; trends are dropped


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, kernel, factor, dropout, c_out):
        super().__init__()
        self.self_correlation = AutoCorrelation(d_model, n_heads, factor)
        self.cross_correlation = AutoCorrelation(d_model, n_heads, factor)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.decomp1, self.decomp2, self.decomp3 = (SeriesDecomposition(kernel) for _ in range(3))
        self.trend_projection = nn.Conv1d(d_model, c_out, kernel_size=3, padding=1,
                                          padding_mode="circular", bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoded):
        x, trend1 = self.decomp1(x + self.dropout(self.self_correlation(x, x)))
        x, trend2 = self.decomp2(x + self.dropout(self.cross_correlation(x, encoded)))
        x, trend3 = self.decomp3(x + self.feed_forward(x))
        residual_trend = self.trend_projection((trend1 + trend2 + trend3).transpose(1, 2)).transpose(1, 2)
        return x, residual_trend


class Embedding(nn.Module):
    """Value embedding (circular Conv1d, k=3) plus an optional linear embedding of per-step covariates."""

    def __init__(self, c_in: int, n_covariates: int, d_model: int, dropout: float, cov_kernel: int = 1):
        super().__init__()
        self.value = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.cov_kernel = cov_kernel
        if not n_covariates:
            self.covariates = None
        elif cov_kernel == 1:                      # like the official TimeFeatureEmbedding: a per-step linear map
            self.covariates = nn.Linear(n_covariates, d_model, bias=False)
        else:                                       # local covariate context; replicate padding never wraps time
            self.covariates = nn.Conv1d(n_covariates, d_model, kernel_size=cov_kernel, padding=cov_kernel // 2,
                                        padding_mode="replicate", bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, marks=None):
        out = self.value(x.transpose(1, 2)).transpose(1, 2)
        if self.covariates is not None:
            if self.cov_kernel == 1:
                out = out + self.covariates(marks)
            else:
                out = out + self.covariates(marks.transpose(1, 2)).transpose(1, 2)
        return self.dropout(out)


class Autoformer(nn.Module):
    def __init__(self, seq_len=168, label_len=84, pred_len=168, c_in=1, c_out=1, n_covariates=0,
                 d_model=32, n_heads=4, e_layers=1, d_layers=1, d_ff=64, kernel=25, factor=1.0, dropout=0.05,
                 cov_kernel=1, anchor=False, trend_init="mean"):
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.anchor = bool(anchor)       # subtract the last observed value from the input and add it back
        self.trend_init = trend_init     # future trend start: "mean" (paper), "last" (end of MA trend), "lastobs",
        # or "decay": mean + (last observation - mean) * rho^h, a learned AR(1)-style reversion (one parameter)
        if trend_init == "decay":
            self.decay_logit = nn.Parameter(torch.tensor(3.0))          # rho = sigmoid(3) = 0.95 at start
        self.decomp = SeriesDecomposition(kernel)
        self.enc_embedding = Embedding(c_in, n_covariates, d_model, dropout, cov_kernel)
        self.dec_embedding = Embedding(c_in, n_covariates, d_model, dropout, cov_kernel)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, kernel, factor, dropout)
                                      for _ in range(e_layers)])
        self.encoder_norm = SeasonalLayerNorm(d_model)
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, kernel, factor, dropout, c_out)
                                      for _ in range(d_layers)])
        self.decoder_norm = SeasonalLayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, marks_enc=None, marks_dec=None):
        """x_enc [B, seq_len, c_in]; marks_* [B, seq_len or label_len + pred_len, n_cov] -> [B, pred_len, c_out]."""
        if self.anchor:
            last = x_enc[:, -1:, :]
            x_enc = x_enc - last
        zeros = torch.zeros(x_enc.shape[0], self.pred_len, x_enc.shape[2], device=x_enc.device, dtype=x_enc.dtype)
        seasonal, trend = self.decomp(x_enc)
        if self.trend_init == "mean":
            start = x_enc.mean(dim=1, keepdim=True)
        elif self.trend_init == "last":
            start = trend[:, -1:]
        elif self.trend_init == "lastobs":
            start = x_enc[:, -1:]
        elif self.trend_init != "decay":
            raise ValueError(self.trend_init)
        if self.trend_init == "decay":
            mean, last = x_enc.mean(dim=1, keepdim=True), x_enc[:, -1:]
            steps = torch.arange(1, self.pred_len + 1, device=x_enc.device, dtype=x_enc.dtype).view(1, -1, 1)
            future = mean + (last - mean) * torch.sigmoid(self.decay_logit) ** steps
        else:
            future = start.expand(-1, self.pred_len, -1)
        trend = torch.cat([trend[:, -self.label_len:], future], dim=1)
        seasonal = torch.cat([seasonal[:, -self.label_len:], zeros], dim=1)

        encoded = self.enc_embedding(x_enc, marks_enc)
        for layer in self.encoder:
            encoded = layer(encoded)
        encoded = self.encoder_norm(encoded)

        x = self.dec_embedding(seasonal, marks_dec)
        for layer in self.decoder:
            x, residual_trend = layer(x, encoded)
            trend = trend + residual_trend
        out = trend + self.projection(self.decoder_norm(x))
        out = out[:, -self.pred_len:]
        return out + last if self.anchor else out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
