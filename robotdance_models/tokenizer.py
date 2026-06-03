"""Motion VQ-VAE — モーションを離散トークン列に符号化・復元する（v0）。

canonical motion window を時間方向にダウンサンプルした潜在列に符号化し、各潜在ベクトルを
学習済み codebook の最近傍コードに**量子化**して離散トークン（コード index）にする。decoder で
元の motion window を復元する。これにより 1 本のモーションが「離散トークンの列」になり、
将来の autoregressive 生成・補完・テキスト条件付け（VLA 接続）の足場になる。

前処理は手作り embedding / contrastive と共有（`robotdance_motion.normalized_keypoints`）。

⚠️ v0: tokenizer 基盤の提供が目的。合成 corpus で再構成 loss が下がり・codebook が使われることを
示すが、**生成 prior（トークン列の言語モデル）は別途**で、本モジュールは符号化⇄復号のみ。
事前学習・実データ規模・residual VQ / 可変長は今後。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from robotdance_core.rd_mir import RdMir, Skeleton
from robotdance_core.skeleton import JOINT_NAMES, NUM_JOINTS, PARENTS
from robotdance_motion.embeddings import normalized_keypoints

from .encoder import INPUT_DIM, window_motion

DOWNSAMPLE = 4  # 時間方向の圧縮率（W frame → W/DOWNSAMPLE token）


class VectorQuantizerEMA(nn.Module):
    """EMA 更新の codebook を持つ vector quantizer（van den Oord 2017）。

    潜在 [.., D] を最近傍コードに量子化し、straight-through で勾配を通す。codebook は
    勾配ではなく指数移動平均で更新する（学習が安定）。commitment loss のみ返す。
    """

    def __init__(self, num_codes: int = 128, dim: int = 64, beta: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        codebook = torch.randn(num_codes, dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", codebook.clone())
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _data_init(self, flat: torch.Tensor) -> None:
        """最初のバッチの encoder 出力から codebook を初期化（collapse 回避）。"""
        n = flat.shape[0]
        sel = torch.randint(0, n, (self.num_codes,), device=flat.device)
        self.codebook.copy_(flat[sel])
        self.ema_w.copy_(self.codebook)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _revive_dead_codes(self, flat: torch.Tensor, threshold: float = 1.0) -> int:
        """使用頻度が低い（cluster_size < threshold）コードを encoder 出力で再初期化する。"""
        dead = self.cluster_size < threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        sel = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        self.codebook[dead] = flat[sel]
        self.ema_w[dead] = flat[sel]
        self.cluster_size[dead] = 1.0
        return n_dead

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        # ||z - e||^2 = ||z||^2 - 2 z·e + ||e||^2
        return (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )

    def forward(self, z: torch.Tensor):
        """z [B, T, D] → (z_q [B, T, D], indices [B, T], commit_loss スカラー)。"""
        b, t, d = z.shape
        flat = z.reshape(-1, d)                       # [N, D]
        if self.training and not bool(self.initialized):
            self._data_init(flat.detach())
        idx = self._distances(flat).argmin(dim=1)     # [N]
        quant = self.codebook[idx].view(b, t, d)

        if self.training:
            self._ema_update(flat, idx)

        commit_loss = self.beta * F.mse_loss(z, quant.detach())
        z_q = z + (quant - z).detach()                # straight-through
        return z_q, idx.view(b, t), commit_loss

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.num_codes).type(flat.dtype)  # [N, K]
        self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
        dw = onehot.t() @ flat                                    # [K, D]
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        cluster = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.codebook.copy_(self.ema_w / cluster.unsqueeze(1))

    @torch.no_grad()
    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """token index [.., T] → コードベクトル [.., T, D]。"""
        return self.codebook[idx]


class MotionVQVAE(nn.Module):
    """motion window [B, W, INPUT_DIM] ⇄ 離散トークン [B, W/DOWNSAMPLE]。"""

    def __init__(self, *, d_model: int = 64, num_codes: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(INPUT_DIM, d_model, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, 4, stride=2, padding=1),
        )
        self.vq = VectorQuantizerEMA(num_codes=num_codes, dim=d_model)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(d_model, INPUT_DIM, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor):
        """x [B, W, INPUT_DIM] → (z_q [B, T, D], indices [B, T], commit_loss)。"""
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D]
        return self.vq(h)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q [B, T, D] → 再構成 motion [B, W, INPUT_DIM]。"""
        return self.decoder(z_q.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor):
        z_q, idx, commit = self.encode(x)
        recon = self.decode(z_q)
        return recon, idx, commit


def _windows_from(motions: list[RdMir], window: int, stride: int) -> np.ndarray:
    chunks = [window_motion(normalized_keypoints(m), window, stride) for m in motions]
    return np.concatenate(chunks, axis=0).astype(np.float32)


def train_tokenizer(
    *,
    out_path: str | Path = "motion_tokenizer.pt",
    window: int = 32,
    stride: int = 8,
    num_codes: int = 128,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 2e-3,
    seed: int = 0,
    device: Optional[str] = None,
) -> dict:
    """合成 corpus で motion VQ-VAE を学習し checkpoint を保存する。

    loss 履歴・最終再構成 MSE・codebook 使用率（学習データで使われたコード割合）を返す。
    """
    from robotdance_models.train import build_corpus

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    data = torch.from_numpy(_windows_from(build_corpus(), window, stride))
    model = MotionVQVAE(num_codes=num_codes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = data.shape[0]
    history: list[float] = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        epoch_loss = 0.0
        for s in range(0, n, batch_size):
            xb = data[perm[s:s + batch_size]].to(dev)
            recon, _, commit = model(xb)
            loss = F.mse_loss(recon, xb) + commit
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        history.append(epoch_loss / n)
        # 序盤に dead code を encoder 出力で復活させて collapse を防ぐ。
        if 0 < ep < epochs // 2 and ep % 5 == 0:
            with torch.no_grad():
                _, _, _ = model(data.to(dev))  # cluster_size を最新化
                flat = model.encoder(data.to(dev).transpose(1, 2)).transpose(1, 2).reshape(-1, model.vq.dim)
                model.vq._revive_dead_codes(flat)

    # 評価: 再構成 MSE と codebook 使用率。
    model.eval()
    with torch.no_grad():
        recon, idx, _ = model(data.to(dev))
        recon_mse = float(F.mse_loss(recon, data.to(dev)).item())
        used = int(torch.unique(idx).numel())
    usage = used / num_codes

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {"window": window, "num_codes": num_codes},
        "loss_history": history,
        "recon_mse": recon_mse,
        "codebook_usage": usage,
    }
    torch.save(ckpt, str(out_path))
    return {
        "loss_history": history,
        "recon_mse": recon_mse,
        "codebook_usage": usage,
        "codes_used": used,
        "num_codes": num_codes,
        "tokens_per_window": window // DOWNSAMPLE,
        "checkpoint": str(out_path),
        "windows": n,
        "device": dev,
    }


class MotionTokenizer:
    """学習済み VQ-VAE。RD-MIR ⇄ 離散トークン列の符号化・復元を提供する。"""

    def __init__(self, checkpoint: str | Path = "motion_tokenizer.pt",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        self.window = ckpt["config"]["window"]
        self.num_codes = ckpt["config"]["num_codes"]
        self.model = MotionVQVAE(num_codes=self.num_codes).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def encode(self, mir: RdMir) -> np.ndarray:
        """RD-MIR を離散トークン列 [n_tokens] に符号化する（非重複タイル）。"""
        rel = normalized_keypoints(mir)
        windows = window_motion(rel, self.window, self.window)  # 非重複
        x = torch.from_numpy(windows.astype(np.float32)).to(self.device)
        _, idx, _ = self.model.encode(x)                        # [n_win, T]
        return idx.reshape(-1).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, mir: RdMir) -> tuple[np.ndarray, np.ndarray]:
        """正規化空間での (元 [F, J, 3], 再構成 [F, J, 3]) を返す（タイル長に丸め）。"""
        rel = normalized_keypoints(mir)
        windows = window_motion(rel, self.window, self.window)
        x = torch.from_numpy(windows.astype(np.float32)).to(self.device)
        recon, _, _ = self.model(x)                             # [n_win, W, INPUT_DIM]
        n_win = windows.shape[0]
        orig = x.cpu().numpy().reshape(n_win * self.window, NUM_JOINTS, 3)
        rec = recon.cpu().numpy().reshape(n_win * self.window, NUM_JOINTS, 3)
        return orig, rec

    @torch.no_grad()
    def decode_to_mir(self, tokens: np.ndarray, *, fps: float = 30.0,
                      motion_id: str = "rdmir-detokenized") -> RdMir:
        """トークン列 → 正規化 keypoints の RD-MIR（codebook lookup → decoder）。"""
        t_per_win = self.window // DOWNSAMPLE
        toks = np.asarray(tokens).reshape(-1, t_per_win)
        idx = torch.from_numpy(toks.astype(np.int64)).to(self.device)
        z_q = self.model.vq.lookup(idx)                         # [n_win, T, D]
        recon = self.model.decode(z_q)                          # [n_win, W, INPUT_DIM]
        n = recon.shape[0] * self.window
        kps = recon.cpu().numpy().reshape(n, NUM_JOINTS, 3)
        return RdMir(
            motion_id=motion_id,
            source_ref={"generator": "robotdance_models.tokenizer.decode"},
            license_state="redistributable",
            fps=fps,
            duration=n / fps,
            skeleton=Skeleton(joint_names=JOINT_NAMES, parents=PARENTS),
            keypoints_3d=kps.tolist(),
            extractor_versions={"tokenizer": "robotdance.vqvae.v0"},
        )
