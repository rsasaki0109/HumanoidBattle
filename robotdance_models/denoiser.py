"""Motion denoiser / in-betweening — VQ-VAE トークン列の masked modeling（§4.2 拡張, v0）。

token prior（`prior.py`）は **causal**（左→右）でモーションを生成・補完する。本モジュールは
**双方向（bidirectional）Transformer** を **masked token modeling**（BERT 風）で学習し、

    denoise()    : 壊れた/外れたトークンを文脈から復元 → ノイズ除去（クリーンな動きに整える）
    inbetween()  : 両端トークンを残し中間を mask → 補間（in-betweening / モーション中割り）

を提供する。causal prior が「続きを作る」のに対し、双方向 denoiser は「全体の文脈で埋める/直す」。
これにより foundation model スタックが **生成（prior）+ 補間/除去（denoiser）** を備える。

⚠️ v0: 小さな合成 corpus で学習。トークン分布に沿う復元・補間を示すが、betas/長尺・実データ規模は
今後。生成物は物理的に妥当とは限らない — retarget → sim_certificate を必ず通すこと。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from robotdance_core.rd_mir import RdMir

from .prior import _token_sequences
from .tokenizer import MotionTokenizer


class MotionDenoiserNet(nn.Module):
    """トークン列 [B, L]（MASK を含む）→ 各位置の code logits [B, L, num_codes]（双方向）。"""

    def __init__(self, *, num_codes: int, d_model: int = 128, nhead: int = 4,
                 nlayers: int = 3, max_len: int = 16) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.mask_id = num_codes              # MASK は num_codes 番（埋め込みのみ、出力には含めない）
        self.max_len = max_len
        self.token_emb = nn.Embedding(num_codes + 1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                           batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, num_codes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        h = self.token_emb(tokens) + self.pos[:, :length]
        h = self.encoder(h)                   # mask なし = 双方向
        return self.head(h)


def train_denoiser(
    *,
    tokenizer_ckpt: str | Path = "motion_tokenizer.pt",
    out_path: str | Path = "motion_denoiser.pt",
    seq_len: int = 16,
    epochs: int = 300,
    batch_size: int = 32,
    mask_ratio: float = 0.3,
    lr: float = 3e-4,
    seed: int = 0,
    device: Optional[str] = None,
) -> dict:
    """masked token modeling で denoiser を学習し checkpoint を保存する。

    各ステップで mask_ratio の位置を選び、BERT 風に 80% を MASK / 10% をランダム置換 / 10% 維持
    として入力し、選んだ位置の元トークンを予測する（loss は選んだ位置のみ）。
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)

    tok = MotionTokenizer(tokenizer_ckpt, device=dev)
    num_codes = tok.num_codes
    mask_id = num_codes

    seqs = torch.from_numpy(_token_sequences(tok, seq_len)).to(dev)  # [N, L] code tokens
    n, length = seqs.shape

    model = MotionDenoiserNet(num_codes=num_codes, max_len=seq_len).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seqs.sum()) + _))
        epoch_loss = 0.0
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            clean = seqs[b]
            bsz = clean.shape[0]
            sel = torch.rand(bsz, length, device=dev, generator=gen) < mask_ratio
            sel[sel.sum(dim=1) == 0, 0] = True   # 各行最低 1 つは mask
            inp = clean.clone()
            r = torch.rand(bsz, length, device=dev, generator=gen)
            mask_now = sel & (r < 0.8)
            rand_now = sel & (r >= 0.8) & (r < 0.9)
            inp[mask_now] = mask_id
            inp[rand_now] = torch.randint(0, num_codes, (int(rand_now.sum()),), device=dev,
                                          generator=gen)
            logits = model(inp)                  # [B, L, num_codes]
            loss = F.cross_entropy(logits[sel], clean[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * bsz
        history.append(epoch_loss / n)

    # masked 位置の復元精度（評価: 一様に mask_ratio を mask）。
    model.eval()
    with torch.no_grad():
        sel = torch.rand(n, length, device=dev, generator=gen) < mask_ratio
        sel[sel.sum(dim=1) == 0, 0] = True
        inp = seqs.clone()
        inp[sel] = mask_id
        pred = model(inp).argmax(dim=-1)
        acc = float((pred[sel] == seqs[sel]).float().mean().item())

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {"num_codes": num_codes, "mask_id": mask_id, "seq_len": seq_len},
        "tokenizer_ckpt": str(tokenizer_ckpt),
        "loss_history": history,
        "masked_token_acc": acc,
    }
    torch.save(ckpt, str(out_path))
    return {
        "loss_history": history,
        "masked_token_acc": acc,
        "checkpoint": str(out_path),
        "sequences": n,
        "seq_len": seq_len,
        "num_codes": num_codes,
        "device": dev,
    }


class MotionDenoiser:
    """tokenizer + 双方向 denoiser。トークン列のノイズ除去 / 補間を行い RD-MIR に decode する。"""

    def __init__(self, denoiser_ckpt: str | Path = "motion_denoiser.pt",
                 tokenizer_ckpt: str | Path | None = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(denoiser_ckpt), map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.num_codes = cfg["num_codes"]
        self.mask_id = cfg["mask_id"]
        self.seq_len = cfg["seq_len"]
        self.model = MotionDenoiserNet(num_codes=self.num_codes, max_len=self.seq_len).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.tok = MotionTokenizer(tokenizer_ckpt or ckpt["tokenizer_ckpt"], device=self.device)

    @torch.no_grad()
    def _fill_windowed(self, cur: np.ndarray, iters: int) -> np.ndarray:
        """seq_len 窓で双方向充填（cur 中の MASK を全て埋める。長い列はスライド窓）。"""
        cur = cur.copy()
        L = len(cur)
        win = self.seq_len
        for _ in range(iters):
            for start in range(0, max(1, L - win + 1), max(1, win // 2)):
                seg = slice(start, min(start + win, L))
                chunk = torch.tensor([cur[seg]], device=self.device)
                pred = self.model(chunk).argmax(dim=-1)[0].cpu().numpy()
                for local, gpos in enumerate(range(seg.start, seg.stop)):
                    if cur[gpos] == self.mask_id:
                        cur[gpos] = int(pred[local])
        # 端の取りこぼし（最後の窓に入らない mask）を最終窓で埋める。
        if (cur == self.mask_id).any():
            seg = slice(max(0, L - win), L)
            chunk = torch.tensor([cur[seg]], device=self.device)
            pred = self.model(chunk).argmax(dim=-1)[0].cpu().numpy()
            for local, gpos in enumerate(range(seg.start, seg.stop)):
                if cur[gpos] == self.mask_id:
                    cur[gpos] = int(pred[local])
        return cur

    @torch.no_grad()
    def _token_logprob(self, ids: np.ndarray) -> np.ndarray:
        """各位置の「現トークンの対数尤度」を双方向モデルで評価する [L]。"""
        L = len(ids)
        win = self.seq_len
        lp = np.full(L, -np.inf)
        for start in range(0, max(1, L - win + 1), max(1, win // 2)):
            seg = slice(start, min(start + win, L))
            chunk = torch.tensor([ids[seg]], device=self.device)
            logp = F.log_softmax(self.model(chunk)[0], dim=-1).cpu().numpy()  # [w, num_codes]
            for local, gpos in enumerate(range(seg.start, seg.stop)):
                lp[gpos] = max(lp[gpos], float(logp[local, ids[gpos]]))
        return lp

    def denoise(self, mir: RdMir, *, detect_ratio: float = 0.2, iters: int = 2,
                fps: float = 30.0) -> tuple[RdMir, dict]:
        """尤度の低いトークンを外れ値とみなし mask→双方向充填でノイズ除去する。"""
        ids = self.tok.encode(mir)
        if len(ids) == 0:
            return self.tok.decode_to_mir(ids, fps=fps, motion_id="rdmir-denoised"), {"masked": 0}
        lp = self._token_logprob(ids)
        k = max(1, int(round(detect_ratio * len(ids))))
        positions = np.argsort(lp)[:k]            # 尤度が低い順に k 個
        restored = self._fill_windowed(_masked(ids, positions, self.mask_id), iters)
        return (self.tok.decode_to_mir(restored, fps=fps, motion_id="rdmir-denoised"),
                {"masked": int(k), "tokens": int(len(ids)),
                 "changed": int((restored != ids).sum())})

    def inbetween(self, mir: RdMir, *, keep: int = 2, iters: int = 3,
                  fps: float = 30.0) -> tuple[RdMir, np.ndarray]:
        """両端 keep トークンを残し中間を mask→双方向充填で補間する。(補間 mir, tokens) を返す。"""
        ids = self.tok.encode(mir)
        positions = np.arange(keep, max(keep, len(ids) - keep))
        restored = self._fill_windowed(_masked(ids, positions, self.mask_id), iters)
        return self.tok.decode_to_mir(restored, fps=fps, motion_id="rdmir-inbetween"), restored


def _masked(ids: np.ndarray, positions: np.ndarray, mask_id: int) -> np.ndarray:
    out = ids.copy().astype(np.int64)
    out[positions] = mask_id
    return out
