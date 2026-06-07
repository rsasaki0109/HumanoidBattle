"""motion embedding / retrieval / Motion Map の検証。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from robotdance_core.synthetic import generate_backflip, generate_dance
from robotdance_motion.embeddings import EMBEDDING_DIM, MotionIndex, embed


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_embedding_dim_and_determinism() -> None:
    mir = generate_dance(duration=2.0)
    e1, e2 = embed(mir), embed(mir)
    assert e1.shape == (EMBEDDING_DIM,)
    np.testing.assert_allclose(e1, e2)


def test_embedding_is_translation_scale_yaw_invariant() -> None:
    """root-relative + scale 正規化 + yaw 整列 → 平行移動/スケール/向きに不変。"""
    mir = generate_dance(duration=2.0)
    base = embed(mir)

    kps = mir.keypoints_3d_array()
    # 平行移動 + 等方スケール + z 軸回り 90° 回転。
    transformed = (Rot.from_euler("z", np.pi / 2).apply(kps.reshape(-1, 3)) * 1.5
                   + np.array([3.0, -2.0, 0.5])).reshape(kps.shape)
    data = mir.to_dict()
    data["keypoints_3d"] = transformed.tolist()
    from robotdance_core.rd_mir import RdMir

    moved = embed(RdMir.model_validate(data))
    assert _cos(base, moved) > 0.99


def test_retrieval_ranks_same_class_higher() -> None:
    idx = MotionIndex()
    for mid, mir in {
        "dance_a": generate_dance(beats_per_second=1.0),
        "dance_b": generate_dance(beats_per_second=1.4),
        "backflip": generate_backflip(),
    }.items():
        mir.motion_id = mid
        idx.add_mir(mir)
    ranked = idx.query(embed(generate_dance(beats_per_second=1.2)), k=3)
    ids = [r[0] for r in ranked]
    # dance 2 件が backflip より上位。
    assert ids.index("backflip") == 2


def test_duplicate_detection() -> None:
    idx = MotionIndex()
    for mid in ("orig", "copy", "other"):
        mir = generate_backflip() if mid == "other" else generate_dance(beats_per_second=1.0)
        mir.motion_id = mid
        idx.add_mir(mir)
    dups = idx.duplicates(threshold=0.98)
    pairs = {frozenset((a, b)) for a, b, _ in dups}
    assert frozenset(("orig", "copy")) in pairs  # 同一 dance は重複検出
    assert frozenset(("orig", "other")) not in pairs


def test_project_2d_separates_classes() -> None:
    idx = MotionIndex()
    for mid, mir in {
        "dance_a": generate_dance(beats_per_second=1.0),
        "dance_b": generate_dance(beats_per_second=1.3),
        "backflip_a": generate_backflip(duration=1.6),
        "backflip_b": generate_backflip(duration=1.4),
    }.items():
        mir.motion_id = mid
        idx.add_mir(mir)
    p = idx.project_2d()
    assert p.shape == (4, 2)
    d_intra = np.linalg.norm(p[0] - p[1])           # dance 同士
    d_inter = np.linalg.norm(p[0] - p[2])           # dance vs backflip
    assert d_inter > d_intra


def test_render_motion_map(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from robotdance_viewer.motion_map import render_motion_map

    pts = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
    out = render_motion_map(pts, ["a", "b", "c"], tmp_path / "map.png",
                            groups=["x", "x", "y"])
    assert out.exists() and out.stat().st_size > 0


def test_query_where_filters_by_metadata() -> None:
    """where 述語でメタデータ絞り込み（quality-aware / label-aware retrieval）。"""
    idx = MotionIndex()
    for mid, hp in [("good_a", "ok"), ("good_b", "ok"), ("bad", "warn")]:
        mir = generate_dance(beats_per_second=1.2)
        mir.motion_id = mid
        idx.add(mid, embed(mir), meta={"health": hp})
    q = embed(generate_dance(beats_per_second=1.2))
    healthy = idx.query(q, k=5, where=lambda m: m.get("health") == "ok")
    ids = [r[0] for r in healthy]
    assert "bad" not in ids
    assert set(ids) == {"good_a", "good_b"}
    # フィルタ無しなら bad も候補に入る。
    assert "bad" in [r[0] for r in idx.query(q, k=5)]


def test_query_text_matches_labels_with_concept_normalization() -> None:
    """query_text: 概念正規化テキスト類似で action_label を引く。未学習の言い換えにも頑健。"""
    idx = MotionIndex()
    labels = {
        "flip1": "an acrobatic backflip",
        "dance1": "a person dancing energetically",
        "idle1": "standing still",
    }
    for mid, lab in labels.items():
        mir = generate_dance(beats_per_second=1.0)
        mir.motion_id = mid
        idx.add(mid, embed(mir), meta={"action_label": lab})

    # 言い換え query が正しい label を top-1 で引く（jog/somersault/upbeat 等は学習不要で概念一致）。
    assert idx.query_text("doing a somersault", k=1)[0][0] == "flip1"
    assert idx.query_text("an upbeat groovy dance", k=1)[0][0] == "dance1"
    assert idx.query_text("a person standing motionless", k=1)[0][0] == "idle1"
    # label を持たない entry は対象外、空 query は空結果。
    idx.add("nolabel", embed(generate_dance()), meta={})
    assert "nolabel" not in [r[0] for r in idx.query_text("dance", k=5)]
    assert idx.query_text("", k=5) == []
    # where 述語で絞り込める。
    hits = idx.query_text("dance", k=5, where=lambda m: m.get("action_label") != labels["dance1"])
    assert "dance1" not in [r[0] for r in hits]


def test_add_mir_diagnose_stores_health() -> None:
    from robotdance_core.synthetic import generate_squat

    idx = MotionIndex()
    mir = generate_squat(duration=1.5)
    mir.motion_id = "sq"
    idx.add_mir(mir, diagnose=True)
    meta = idx.meta_of("sq")
    assert meta["health"] in ("ok", "warn", "unknown")
    assert "action_label" in meta


def test_cli_search_motion_healthy_only(tmp_path) -> None:
    from robotdance_core.cli import main
    from robotdance_core.synthetic import generate_dance

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(3):
        m = generate_dance(beats_per_second=1.0 + 0.2 * i)
        m.motion_id = f"d{i}"
        m.save(corpus / f"d{i}.json")
    q = generate_dance(beats_per_second=1.1)
    q.save(tmp_path / "q.json")

    assert main(["search-motion", str(tmp_path / "q.json"), str(corpus), "-k", "2"]) == 0
    assert main(["search-motion", str(tmp_path / "q.json"), str(corpus),
                 "--healthy-only"]) == 0


def test_cli_search_motion_by_text(tmp_path, capsys) -> None:
    """CLI search-motion --text: 自然言語で action_label を概念照合して引く（学習不要）。"""
    from robotdance_core.cli import main
    from robotdance_core.synthetic import generate_backflip, generate_dance

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    specs = [("flip", generate_backflip(duration=1.4), "an acrobatic backflip"),
             ("groove", generate_dance(beats_per_second=1.4), "energetic dancing")]
    for mid, m, label in specs:
        m.motion_id = mid
        m.semantics = {**(m.semantics or {}), "action_label": label}
        m.save(corpus / f"{mid}.json")

    # query 位置引数なしで --text のみ。言い換え "somersault" が backflip を引く。
    assert main(["search-motion", str(corpus), "--text", "doing a somersault", "-k", "2"]) == 0
    out = capsys.readouterr().out
    # 上位行に flip が含まれる（概念正規化で somersault→flip 一致）。
    assert "flip" in out
    # query も --text も無ければ rc=2。
    assert main(["search-motion", str(corpus)]) == 2


def test_cli_search_motion_with_learned_encoder(tmp_path) -> None:
    """search-motion --encoder で学習済み encoder を使って検索できる。torch 無しは skip。"""
    pytest.importorskip("torch")
    from robotdance_core.cli import main
    from robotdance_core.synthetic import generate_dance
    from robotdance_models.train import train_encoder

    ckpt = tmp_path / "enc.pt"
    train_encoder(out_path=ckpt, epochs=15, seed=0)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(3):
        m = generate_dance(beats_per_second=1.0 + 0.2 * i)
        m.motion_id = f"d{i}"
        m.save(corpus / f"d{i}.json")
    q = generate_dance(beats_per_second=1.1)
    q.save(tmp_path / "q.json")

    assert main(["search-motion", str(tmp_path / "q.json"), str(corpus),
                 "--encoder", str(ckpt), "-k", "2"]) == 0
