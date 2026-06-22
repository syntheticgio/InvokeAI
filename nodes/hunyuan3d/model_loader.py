from __future__ import annotations

from pathlib import Path
from typing import Any

# Module-level cache: model_path (str) → loaded pipeline object
_MODEL_CACHE: dict[str, Any] = {}


def load_model(model_path: str) -> Any:
    """Return a cached Hunyuan3D pipeline, loading from disk on first call."""
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = _load_from_disk(model_path)
    return _MODEL_CACHE[model_path]


def _load_from_disk(model_path: str) -> Any:
    """Load Hunyuan3D weights from disk using hy3dgen.

    Raises FileNotFoundError if model_path does not exist.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Hunyuan3D model not found at '{model_path}'. "
            f"Download weights with: "
            f"huggingface-cli download tencent/Hunyuan3D-2 "
            f"--local-dir {model_path}"
        )

    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "hy3dgen is not installed. Run: pip install -r nodes/hunyuan3d/requirements.txt"
        ) from e

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    return pipeline
