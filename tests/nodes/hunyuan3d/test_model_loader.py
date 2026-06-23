import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_load_model_returns_cached_instance(tmp_path: Path) -> None:
    """Calling load_model twice with the same path returns the same object."""
    from nodes.hunyuan3d.model_loader import load_model, _MODEL_CACHE

    _MODEL_CACHE.clear()
    mock_pipeline = MagicMock()

    with patch("nodes.hunyuan3d.model_loader._load_from_disk", return_value=mock_pipeline) as mock_load:
        result1 = load_model(str(tmp_path))
        result2 = load_model(str(tmp_path))

    assert result1 is result2
    mock_load.assert_called_once_with(str(tmp_path))


def test_load_model_different_paths_different_instances(tmp_path: Path) -> None:
    """Different model paths produce different cached entries."""
    from nodes.hunyuan3d.model_loader import load_model, _MODEL_CACHE

    _MODEL_CACHE.clear()
    mock1, mock2 = MagicMock(), MagicMock()
    call_count = {"n": 0}

    def fake_load(path: str):
        call_count["n"] += 1
        return mock1 if call_count["n"] == 1 else mock2

    with patch("nodes.hunyuan3d.model_loader._load_from_disk", side_effect=fake_load):
        r1 = load_model(str(tmp_path / "model_a"))
        r2 = load_model(str(tmp_path / "model_b"))

    assert r1 is not r2


def test_load_from_disk_raises_if_path_missing(tmp_path: Path) -> None:
    """_load_from_disk raises FileNotFoundError when the path does not exist."""
    from nodes.hunyuan3d.model_loader import _load_from_disk

    with pytest.raises(FileNotFoundError, match="not found"):
        _load_from_disk(str(tmp_path / "nonexistent"))


def test_load_from_disk_raises_on_missing_hy3dgen(tmp_path: Path) -> None:
    """_load_from_disk raises ImportError with helpful message when hy3dgen is not installed."""
    from nodes.hunyuan3d.model_loader import _load_from_disk

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Simulate hy3dgen not being installed by replacing it with None in sys.modules
    with patch.dict(sys.modules, {"hy3dgen": None, "hy3dgen.shapegen": None}):
        with pytest.raises(ImportError, match="hy3dgen is not installed"):
            _load_from_disk(str(model_dir))


def test_load_from_disk_passes_detected_device_to_pipeline(tmp_path: Path) -> None:
    """_load_from_disk passes InvokeAI's chosen torch device to from_pretrained,
    instead of relying on hy3dgen's hardcoded 'cuda' default (which crashes on
    machines without CUDA, e.g. Apple Silicon)."""
    import torch

    from nodes.hunyuan3d.model_loader import _load_from_disk

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    fake_pipeline_cls = MagicMock()
    fake_module = MagicMock()
    fake_module.Hunyuan3DDiTFlowMatchingPipeline = fake_pipeline_cls

    with patch.dict(sys.modules, {"hy3dgen.shapegen": fake_module}), \
         patch(
             "invokeai.backend.util.devices.TorchDevice.choose_torch_device",
             return_value=torch.device("mps"),
         ):
        _load_from_disk(str(model_dir))

    fake_pipeline_cls.from_pretrained.assert_called_once_with(str(model_dir), device="mps")
