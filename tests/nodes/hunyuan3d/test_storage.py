import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_save_mesh_creates_directory(tmp_path: Path) -> None:
    """save_mesh creates the 3d subdirectory if it doesn't exist."""
    from nodes.hunyuan3d.storage import save_mesh

    mesh_bytes = b"fake mesh content"
    with patch("nodes.hunyuan3d.storage._get_outputs_3d_dir", return_value=tmp_path / "3d"):
        result_path = save_mesh(mesh_bytes, "glb")

    assert (tmp_path / "3d").exists()
    assert result_path.endswith(".glb")
    assert os.path.isabs(result_path)


def test_save_mesh_writes_content(tmp_path: Path) -> None:
    """save_mesh writes the exact bytes provided."""
    from nodes.hunyuan3d.storage import save_mesh

    mesh_bytes = b"fake glb content xyz"
    with patch("nodes.hunyuan3d.storage._get_outputs_3d_dir", return_value=tmp_path / "3d"):
        result_path = save_mesh(mesh_bytes, "glb")

    assert Path(result_path).read_bytes() == mesh_bytes


def test_save_mesh_supports_obj(tmp_path: Path) -> None:
    """save_mesh uses the correct extension for OBJ format."""
    from nodes.hunyuan3d.storage import save_mesh

    with patch("nodes.hunyuan3d.storage._get_outputs_3d_dir", return_value=tmp_path / "3d"):
        result_path = save_mesh(b"obj data", "obj")

    assert result_path.endswith(".obj")


def test_save_mesh_unique_filenames(tmp_path: Path) -> None:
    """Two calls produce different filenames."""
    from nodes.hunyuan3d.storage import save_mesh

    with patch("nodes.hunyuan3d.storage._get_outputs_3d_dir", return_value=tmp_path / "3d"):
        path1 = save_mesh(b"data1", "glb")
        path2 = save_mesh(b"data2", "glb")

    assert path1 != path2
