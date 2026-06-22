from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def _make_mock_context(tmp_path: Path) -> MagicMock:
    """Build a minimal fake InvocationContext."""
    ctx = MagicMock()
    ctx.images.get_pil.return_value = Image.new("RGB", (64, 64), color=(128, 64, 32))
    fake_dto = MagicMock()
    fake_dto.image_name = "test-thumb-001.png"
    fake_dto.width = 256
    fake_dto.height = 256
    ctx.images.save.return_value = fake_dto
    return ctx


def test_image_to_3d_returns_hunyuan3d_output(tmp_path: Path) -> None:
    """ImageTo3DInvocation.invoke returns a Hunyuan3DOutput with mesh_path and thumbnail."""
    from nodes.hunyuan3d.invocations import ImageTo3DInvocation, Hunyuan3DOutput
    from invokeai.app.invocations.fields import ImageField

    invocation = ImageTo3DInvocation(
        id="test-node-1",
        image=ImageField(image_name="source.png"),
        model_path=str(tmp_path / "fake_model"),
        steps=10,
        guidance_scale=5.0,
        output_format="glb",
        render_thumbnail=True,
    )

    fake_mesh_bytes = b"FAKE_GLB_CONTENT"
    fake_mesh = MagicMock()
    fake_mesh.export.return_value = fake_mesh_bytes

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [fake_mesh]

    ctx = _make_mock_context(tmp_path)

    with patch("nodes.hunyuan3d.invocations.load_model", return_value=fake_pipeline), \
         patch("nodes.hunyuan3d.invocations.save_mesh", return_value=str(tmp_path / "out.glb")) as mock_save, \
         patch("nodes.hunyuan3d.invocations._render_thumbnail", return_value=Image.new("RGB", (256, 256))):
        result = invocation.invoke(ctx)

    assert isinstance(result, Hunyuan3DOutput)
    assert result.mesh_path == str(tmp_path / "out.glb")
    assert result.format == "glb"
    assert result.thumbnail.image_name == "test-thumb-001.png"
    mock_save.assert_called_once_with(fake_mesh_bytes, "glb")


def test_image_to_3d_skips_thumbnail_when_disabled(tmp_path: Path) -> None:
    """When render_thumbnail=False, context.images.save is never called."""
    from nodes.hunyuan3d.invocations import ImageTo3DInvocation
    from invokeai.app.invocations.fields import ImageField

    invocation = ImageTo3DInvocation(
        id="test-node-2",
        image=ImageField(image_name="source.png"),
        model_path=str(tmp_path / "fake_model"),
        steps=10,
        guidance_scale=5.0,
        output_format="glb",
        render_thumbnail=False,
    )

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [MagicMock(export=MagicMock(return_value=b"glb"))]

    ctx = _make_mock_context(tmp_path)

    with patch("nodes.hunyuan3d.invocations.load_model", return_value=fake_pipeline), \
         patch("nodes.hunyuan3d.invocations.save_mesh", return_value=str(tmp_path / "out.glb")):
        result = invocation.invoke(ctx)

    ctx.images.save.assert_not_called()
    assert result.thumbnail is None


def test_image_to_3d_raises_on_missing_model(tmp_path: Path) -> None:
    """invoke raises FileNotFoundError when model_path does not exist."""
    from nodes.hunyuan3d.invocations import ImageTo3DInvocation
    from invokeai.app.invocations.fields import ImageField

    invocation = ImageTo3DInvocation(
        id="test-node-3",
        image=ImageField(image_name="source.png"),
        model_path=str(tmp_path / "nonexistent_model"),
        steps=10,
        guidance_scale=5.0,
        output_format="glb",
        render_thumbnail=True,
    )

    ctx = _make_mock_context(tmp_path)

    with patch("nodes.hunyuan3d.invocations.load_model",
               side_effect=FileNotFoundError("Hunyuan3D model not found")):
        with pytest.raises(FileNotFoundError, match="not found"):
            invocation.invoke(ctx)


def test_image_to_3d_raises_on_empty_pipeline_result(tmp_path: Path) -> None:
    """invoke raises RuntimeError when pipeline returns an empty list."""
    from nodes.hunyuan3d.invocations import ImageTo3DInvocation
    from invokeai.app.invocations.fields import ImageField

    invocation = ImageTo3DInvocation(
        id="test-node-4",
        image=ImageField(image_name="source.png"),
        model_path=str(tmp_path / "fake_model"),
        steps=10,
        guidance_scale=5.0,
        output_format="glb",
        render_thumbnail=False,
    )

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = []  # empty result

    ctx = _make_mock_context(tmp_path)

    with patch("nodes.hunyuan3d.invocations.load_model", return_value=fake_pipeline):
        with pytest.raises(RuntimeError, match="no meshes"):
            invocation.invoke(ctx)


def test_image_to_3d_raises_on_null_export(tmp_path: Path) -> None:
    """invoke raises RuntimeError when mesh.export() returns None."""
    from nodes.hunyuan3d.invocations import ImageTo3DInvocation
    from invokeai.app.invocations.fields import ImageField

    invocation = ImageTo3DInvocation(
        id="test-node-5",
        image=ImageField(image_name="source.png"),
        model_path=str(tmp_path / "fake_model"),
        steps=10,
        guidance_scale=5.0,
        output_format="glb",
        render_thumbnail=False,
    )

    fake_mesh = MagicMock()
    fake_mesh.export.return_value = None  # degenerate mesh

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [fake_mesh]

    ctx = _make_mock_context(tmp_path)

    with patch("nodes.hunyuan3d.invocations.load_model", return_value=fake_pipeline):
        with pytest.raises(RuntimeError, match="export"):
            invocation.invoke(ctx)


def test_text_to_3d_returns_hunyuan3d_output(tmp_path: Path) -> None:
    """TextTo3DInvocation.invoke returns a Hunyuan3DOutput."""
    from nodes.hunyuan3d.invocations import TextTo3DInvocation, Hunyuan3DOutput

    invocation = TextTo3DInvocation(
        id="test-node-6",
        prompt="a wooden barrel",
        model_path=str(tmp_path / "fake_model"),
        steps=10,
        guidance_scale=5.0,
        output_format="glb",
        render_thumbnail=True,
    )

    fake_mesh_bytes = b"FAKE_GLB_TEXT"
    fake_mesh = MagicMock()
    fake_mesh.export.return_value = fake_mesh_bytes

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [fake_mesh]

    ctx = _make_mock_context(tmp_path)

    with patch("nodes.hunyuan3d.invocations.load_model", return_value=fake_pipeline), \
         patch("nodes.hunyuan3d.invocations.save_mesh", return_value=str(tmp_path / "out.glb")), \
         patch("nodes.hunyuan3d.invocations._render_thumbnail", return_value=Image.new("RGB", (256, 256))):
        result = invocation.invoke(ctx)

    assert isinstance(result, Hunyuan3DOutput)
    assert result.format == "glb"


def test_text_to_3d_passes_prompt_to_pipeline(tmp_path: Path) -> None:
    """TextTo3DInvocation passes the prompt string to the pipeline."""
    from nodes.hunyuan3d.invocations import TextTo3DInvocation

    invocation = TextTo3DInvocation(
        id="test-node-7",
        prompt="a glowing crystal orb",
        model_path=str(tmp_path / "fake_model"),
        steps=10,
        guidance_scale=5.0,
        output_format="obj",
        render_thumbnail=False,
    )

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [MagicMock(export=MagicMock(return_value=b"obj data"))]
    ctx = _make_mock_context(tmp_path)

    with patch("nodes.hunyuan3d.invocations.load_model", return_value=fake_pipeline), \
         patch("nodes.hunyuan3d.invocations.save_mesh", return_value=str(tmp_path / "out.obj")):
        invocation.invoke(ctx)

    call_kwargs = fake_pipeline.call_args.kwargs
    assert call_kwargs.get("prompt") == "a glowing crystal orb"


def test_render_thumbnail_raises_on_none_save_image(tmp_path: Path) -> None:
    """_render_thumbnail raises RuntimeError when scene.save_image returns None (headless env)."""
    from nodes.hunyuan3d.invocations import _render_thumbnail

    # Build a real class so that isinstance() works correctly in the function.
    class FakeScene:
        def __init__(self, meshes=None):
            pass

        def save_image(self, resolution=None):
            return None  # simulate headless / no render backend

    mock_trimesh = MagicMock()
    mock_trimesh.Scene = FakeScene

    # Pass a plain object (not a FakeScene) so the else-branch wraps it in FakeScene([mesh]).
    with patch.dict("sys.modules", {"trimesh": mock_trimesh}):
        with pytest.raises(RuntimeError, match="save_image"):
            _render_thumbnail(MagicMock())
