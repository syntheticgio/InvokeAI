import uuid
from pathlib import Path

from invokeai.app.services.config import get_config


def _get_outputs_3d_dir() -> Path:
    outputs_path = get_config().outputs_path
    assert outputs_path is not None, "InvokeAI outputs_path is not configured"
    dir_ = outputs_path / "3d"
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_


def save_mesh(mesh_bytes: bytes, fmt: str) -> str:
    """Write mesh bytes to <outputs_path>/3d/<uuid>.<fmt> and return the absolute path."""
    if not fmt.isalnum():
        raise ValueError(f"Invalid output format '{fmt}': must be alphanumeric (e.g. glb, obj, fbx)")
    dir_ = _get_outputs_3d_dir()
    filename = f"{uuid.uuid4()}.{fmt}"
    dest = dir_ / filename
    dest.write_bytes(mesh_bytes)
    return str(dest.resolve())
