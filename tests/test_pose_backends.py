"""pose 検出バックエンドのレジストリ（robotdance_perception.backends）。"""

from __future__ import annotations

import pytest

from robotdance_perception.backends import (
    MEDIAPIPE,
    get_backend,
    list_backends,
    resolve_extract_backend,
)


def test_registry_lists_known_backends_sorted():
    names = [b.name for b in list_backends()]
    assert names == sorted(names)
    assert {"mediapipe", "yolo11-pose", "rtmpose"} <= set(names)


def test_get_backend_unknown_raises_with_candidates():
    with pytest.raises(ValueError, match="未知の pose backend"):
        get_backend("openpose")


def test_mediapipe_is_3d_and_retarget_capable():
    assert MEDIAPIPE.output_dim == 3
    assert MEDIAPIPE.retarget_capable is True
    assert MEDIAPIPE.keypoint_format == "blazepose33"
    # 公式 Apache-2.0 モデルで dev-only 印は付けない。
    assert "dev" not in MEDIAPIPE.extras


def test_2d_backends_are_not_retarget_capable():
    for name in ("yolo11-pose", "rtmpose"):
        b = get_backend(name)
        assert b.output_dim == 2
        assert b.retarget_capable is False
        assert b.keypoint_format == "coco17"
        assert "dev" in b.extras


def test_resolve_extract_accepts_3d_backend():
    assert resolve_extract_backend("mediapipe") is MEDIAPIPE


def test_resolve_extract_rejects_2d_backend():
    with pytest.raises(ValueError, match="3D が必要"):
        resolve_extract_backend("yolo11-pose")


def test_available_is_boolean_without_importing_heavy_deps():
    # available() は遅延 spec チェックのみ。例外を投げず bool を返す。
    for b in list_backends():
        assert isinstance(b.available(), bool)
