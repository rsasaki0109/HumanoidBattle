"""Motion token prior — VQ-VAE トークン列の autoregressive 生成モデル（v0）。

`tokenizer.py` の VQ-VAE が motion を離散トークン列にする。本モジュールはそのトークン列上で
小さな **causal Transformer**（GPT 風）を next-token 予測で学習し、

    generate()  : BOS から自己回帰サンプリング → トークン列 → VQ-VAE decode → 新規モーション
    complete()  : 既存モーションの prefix トークンに続きを生成 → 補完モーション

を提供する。tokenizer（符号化⇄復号）と prior（トークンの並びの確率モデル）が揃って初めて
「モーション生成・補完」が動く。

⚠️ v0: 小さな合成 corpus で学習。トークンの並びの妥当性（学習分布に沿う滑らかな動き）を示すが、
多様性・新規性・テキスト条件付けは今後。**生成物は物理的に妥当とは限らない** —
retarget → sim_certificate（MuJoCo）の安全パイプラインを必ず通すこと。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from robotdance_core.rd_mir import RdMir

from .tokenizer import MotionTokenizer


class MotionPrior(nn.Module):
    """トークン列 [B, L] → 次トークン logits [B, L, vocab]（causal self-attention）。"""

    def __init__(self, *, vocab: int, d_model: int = 128, nhead: int = 4,
                 nlayers: int = 3, max_len: int = 64) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                           batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, length = tokens.shape
        h = self.token_emb(tokens) + self.pos[:, :length]
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool),
                          diagonal=1)  # 未来を見せない causal mask
        h = self.encoder(h, mask=mask)
        return self.head(h)


def _build_prior_corpus():
    """prior 学習用に、合成 corpus を tempo/振幅で増やした motion 集合（決定的）。"""
    from robotdance_core.synthetic import generate_backflip, generate_dance

    motions = []
    for bps in (0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6):
        motions.append(generate_dance(beats_per_second=bps))
    for arm in (0.1, 0.15, 0.2, 0.25, 0.3):
        motions.append(generate_dance(beats_per_second=0.5, arm_amp=arm, sway_amp=0.05))
    for dur in (1.3, 1.5, 1.7, 1.9):
        motions.append(generate_backflip(duration=dur))
    return motions


def _token_sequences(tok: MotionTokenizer, seq_len: int) -> np.ndarray:
    """corpus を tokenize し、長さ seq_len の固定長サンプル群 [N, seq_len] を作る。

    各 motion の非重複トークン列に対し seq_len のスライディング窓を取る（短ければ末尾値で pad）。
    """
    samples: list[np.ndarray] = []
    for m in _build_prior_corpus():
        ids = tok.encode(m)
        if len(ids) >= seq_len:
            for s in range(0, len(ids) - seq_len + 1):
                samples.append(ids[s:s + seq_len])
        else:
            pad = np.concatenate([ids, np.repeat(ids[-1:], seq_len - len(ids))])
            samples.append(pad)
    return np.stack(samples).astype(np.int64)


def train_prior(
    *,
    tokenizer_ckpt: str | Path = "motion_tokenizer.pt",
    out_path: str | Path = "motion_prior.pt",
    seq_len: int = 16,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 0,
    device: Optional[str] = None,
) -> dict:
    """tokenizer のトークン列で next-token 予測の prior を学習し checkpoint を保存する。

    BOS（id=num_codes）を先頭に付け、各位置で次トークンを予測する。loss 履歴と最終
    next-token 精度（teacher forcing）を返す。
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    tok = MotionTokenizer(tokenizer_ckpt, device=dev)
    num_codes = tok.num_codes
    bos = num_codes
    vocab = num_codes + 1

    seqs = torch.from_numpy(_token_sequences(tok, seq_len)).to(dev)  # [N, L]
    n = seqs.shape[0]
    bos_col = torch.full((n, 1), bos, dtype=torch.long, device=dev)
    inp = torch.cat([bos_col, seqs[:, :-1]], dim=1)                  # [BOS, t0..t_{L-2}]
    tgt = seqs                                                        # [t0..t_{L-1}]

    model = MotionPrior(vocab=vocab, max_len=seq_len).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        epoch_loss = 0.0
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            logits = model(inp[b])                                   # [B, L, vocab]
            loss = F.cross_entropy(logits.reshape(-1, vocab), tgt[b].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * b.shape[0]
        history.append(epoch_loss / n)

    model.eval()
    with torch.no_grad():
        pred = model(inp).argmax(dim=-1)
        acc = float((pred == tgt).float().mean().item())

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {"vocab": vocab, "num_codes": num_codes, "bos": bos, "seq_len": seq_len},
        "tokenizer_ckpt": str(tokenizer_ckpt),
        "loss_history": history,
        "next_token_acc": acc,
    }
    torch.save(ckpt, str(out_path))
    return {
        "loss_history": history,
        "next_token_acc": acc,
        "checkpoint": str(out_path),
        "sequences": n,
        "seq_len": seq_len,
        "vocab": vocab,
        "device": dev,
    }


class MotionGenerator:
    """tokenizer + prior。トークン列を生成/補完し RD-MIR に decode する。"""

    def __init__(self, prior_ckpt: str | Path = "motion_prior.pt",
                 tokenizer_ckpt: str | Path | None = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(prior_ckpt), map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.num_codes = cfg["num_codes"]
        self.bos = cfg["bos"]
        self.seq_len = cfg["seq_len"]
        self.model = MotionPrior(vocab=cfg["vocab"], max_len=self.seq_len).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.tok = MotionTokenizer(tokenizer_ckpt or ckpt["tokenizer_ckpt"], device=self.device)

    @torch.no_grad()
    def _sample(self, prefix: list[int], length: int, temperature: float,
                generator: torch.Generator) -> np.ndarray:
        """[BOS]+prefix から length 個の code token を自己回帰サンプリングする。"""
        seq = [self.bos] + list(prefix)
        while len(seq) - 1 < length:
            ctx = torch.tensor([seq[-self.seq_len:]], device=self.device)
            logits = self.model(ctx)[0, -1]               # [vocab]
            logits[self.bos] = float("-inf")              # BOS は再生成しない
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator).item())
            seq.append(nxt)
        return np.array(seq[1:length + 1], dtype=np.int64)

    def generate(self, *, length: int | None = None, temperature: float = 1.0,
                 seed: int = 0, fps: float = 30.0) -> RdMir:
        """新規モーションを生成する（BOS から）。length は code token 数（既定 seq_len）。"""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        length = length or self.seq_len
        tokens = self._sample([], length, temperature, gen)
        return self.tok.decode_to_mir(tokens, fps=fps, motion_id="rdmir-generated")

    def complete(self, mir: RdMir, *, keep: int = 4, temperature: float = 1.0,
                 seed: int = 0, fps: float = 30.0) -> tuple[RdMir, np.ndarray]:
        """mir の先頭 keep トークンを残し、続きを生成して補完する。(補完 mir, tokens) を返す。"""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        ids = self.tok.encode(mir)
        length = max(len(ids), self.seq_len)
        prefix = ids[:keep].tolist()
        tokens = self._sample(prefix, length, temperature, gen)
        return self.tok.decode_to_mir(tokens, fps=fps, motion_id="rdmir-completed"), tokens
