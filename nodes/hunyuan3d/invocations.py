from pathlib import Path
from typing import Literal, Optional

from PIL import Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    ImageField,
    InputField,
    OutputField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.services.shared.invocation_context import InvocationContext

from .model_loader import load_model
from .storage import save_mesh


@invocation_output("hunyuan3d_output")
class Hunyuan3DOutput(BaseInvocationOutput):
    """Output from Hunyuan3D mesh generation."""

    thumbnail: Optional[ImageField] = OutputField(
        default=None,
        description="Rendered PNG preview of the mesh (None if render_thumbnail=False)",
    )
    mesh_path: str = OutputField(
        description="Absolute filesystem path to the generated 3D file",
    )
    format: str = OutputField(description="Output format: glb | obj | fbx")
    file_size_bytes: int = OutputField(description="Size of the generated mesh file in bytes")


def _render_thumbnail(mesh: object, size: int = 256) -> Image.Image:
    """Render a top-down PNG thumbnail of the mesh using trimesh."""
    from io import BytesIO

    import trimesh  # type: ignore[import]

    scene = mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene([mesh])
    png_bytes = scene.save_image(resolution=(size, size))
    if png_bytes is None:
        raise RuntimeError(
            "trimesh.Scene.save_image() returned None. "
            "A render backend (pyrender or pyglet) must be installed and a display must be available. "
            "Set render_thumbnail=False to skip thumbnail generation."
        )
    return Image.open(BytesIO(png_bytes)).convert("RGB")


@invocation(
    "image_to_3d",
    title="Image to 3D (Hunyuan3D)",
    tags=["3d", "hunyuan3d", "mesh"],
    category="3d",
    version="1.0.0",
    classification=Classification.Beta,
)
class ImageTo3DInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generate a 3D mesh from an input image using Hunyuan3D 2.1."""

    image: ImageField = InputField(description="Source image to convert to 3D")
    model_path: str = InputField(
        default="models/hunyuan3d/hunyuan3d-2-1",
        description="Path to Hunyuan3D weights, relative to InvokeAI root or absolute",
    )
    steps: int = InputField(default=30, ge=10, le=100, description="Number of diffusion steps")
    guidance_scale: float = InputField(default=5.0, ge=1.0, le=20.0, description="Guidance scale")
    output_format: Literal["glb", "obj", "fbx"] = InputField(
        default="glb", description="Output mesh format"
    )
    render_thumbnail: bool = InputField(
        default=True, description="Render a PNG preview thumbnail for the gallery"
    )

    def invoke(self, context: InvocationContext) -> Hunyuan3DOutput:
        # 0. Resolve model path (relative paths are resolved against InvokeAI root)
        resolved_model_path = Path(self.model_path)
        if not resolved_model_path.is_absolute():
            from invokeai.app.services.config import get_config
            resolved_model_path = get_config().root_path / resolved_model_path
        model_path = str(resolved_model_path)

        # 1. Load source image
        source_image: Image.Image = context.images.get_pil(self.image.image_name)

        # 2. Load (or retrieve cached) pipeline
        pipeline = load_model(model_path)

        # 3. Run inference — pipeline returns a list; first element is the mesh
        meshes = pipeline(
            image=source_image,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
        )
        if not meshes:
            raise RuntimeError("Hunyuan3D pipeline returned no meshes. Check input image and model.")
        mesh = meshes[0]

        # 4. Export to bytes and save to disk
        fmt = self.output_format
        raw_export = mesh.export(file_type=fmt)
        mesh_bytes = raw_export.encode("utf-8") if isinstance(raw_export, str) else raw_export
        if not mesh_bytes:
            raise RuntimeError(
                f"mesh.export() returned empty/None for format '{fmt}'. Mesh may be degenerate."
            )
        mesh_path = save_mesh(mesh_bytes, fmt)

        # 5. Optionally render thumbnail
        thumbnail_field: Optional[ImageField] = None
        if self.render_thumbnail:
            thumb_image = _render_thumbnail(mesh)
            image_dto = context.images.save(image=thumb_image)
            thumbnail_field = ImageField(image_name=image_dto.image_name)

        return Hunyuan3DOutput(
            thumbnail=thumbnail_field,
            mesh_path=mesh_path,
            format=fmt,
            file_size_bytes=len(mesh_bytes),
        )
