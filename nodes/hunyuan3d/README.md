# Hunyuan3D Node Pack for InvokeAI

Adds an `Image to 3D (Hunyuan3D)` node to the InvokeAI Workflows tab, powered by [Hunyuan3D 2.1](https://github.com/Tencent/Hunyuan3D-2).

Hunyuan3D 2.1 is an image-to-mesh model — it has no built-in text-to-image stage. To generate a 3D mesh from a text prompt, chain InvokeAI's own text-to-image node in front of this node (see "Generating from a text prompt" below).

## Installation

### 1. Install Python dependencies

```bash
pip install -r nodes/hunyuan3d/requirements.txt
```

### 2. Download model weights

```bash
huggingface-cli download tencent/Hunyuan3D-2 --local-dir models/hunyuan3d/hunyuan3d-2-1
```

This downloads ~10 GB of weights into `models/hunyuan3d/hunyuan3d-2-1/`.

### 3. Restart InvokeAI

The nodes auto-load on startup. Open the **Workflows** tab and search for "3d" in the node palette.

---

## Node: Image to 3D

Converts an existing image into a 3D mesh.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | ImageField | — | Source image (connect from Load Image or any image node) |
| `model_path` | string | `models/hunyuan3d/hunyuan3d-2-1` | Path to weights |
| `steps` | int | 30 | Diffusion steps (10–100) |
| `guidance_scale` | float | 5.0 | Guidance scale (1.0–20.0) |
| `output_format` | glb/obj/fbx | glb | Mesh file format |
| `render_thumbnail` | bool | true | Save a PNG preview to the gallery |

| Output | Type | Description |
|--------|------|-------------|
| `thumbnail` | ImageField | Gallery preview (null if render_thumbnail=false) |
| `mesh_path` | string | Absolute path to the generated mesh file |
| `format` | string | Format of the output file |
| `file_size_bytes` | int | File size in bytes |

### Example workflow

```
[Load Image] → image → [Image to 3D] → thumbnail → [Preview Image]
                                      → mesh_path → [Show Text]
```

---

## Generating from a text prompt

Hunyuan3D only accepts an image, so to go from text to 3D, chain InvokeAI's own image generation node in front of `Image to 3D`:

```
[Text to Image (e.g. SDXL)] → image → [Image to 3D] → thumbnail → [Preview Image]
                                                      → mesh_path → [Show Text]
```

Tips for better meshes:
- Use a prompt that produces a single centered object on a plain background (e.g. "a wooden treasure chest, white background, product photo")
- A square aspect ratio (1:1) works best as input to Hunyuan3D

---

## Output files

Generated mesh files are saved to `<InvokeAI outputs directory>/3d/`. The `mesh_path` output shows the exact absolute path.

Open the file in:
- **Blender**: File → Import → glTF 2.0 (for GLB)
- **Windows 3D Viewer**: double-click a `.glb` file
- **Browser**: drag a `.glb` into https://gltf-viewer.donmccurdy.com

---

## Upgrading InvokeAI

This node pack makes **zero changes** to InvokeAI core files. You can safely run `git pull origin main` at any time without merge conflicts.

## Troubleshooting

**"Hunyuan3D model not found"** — run the `huggingface-cli download` command above and ensure the path matches the `model_path` input.

**"hy3dgen is not installed"** — run `pip install -r nodes/hunyuan3d/requirements.txt`.

**Thumbnail is blank or black** — trimesh's renderer requires a display. On headless servers, install `xvfb` and run InvokeAI with `xvfb-run`. Alternatively, set `render_thumbnail=False` to skip thumbnail generation.
