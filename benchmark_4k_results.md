# CorridorKey 4K Benchmark — Tears of Steel

## Test Configuration

- **Source footage**: Tears of Steel (scene 02_3c) — CC-BY 3.0 (c) Blender Foundation | mango.blender.org
- **Format**: OpenEXR 16-bit half-float, linear color space
- **Resolution**: 4096x2160 (DCI 4K)
- **Frames**: 100 (24 fps, ~4.2 seconds)
- **Model input size**: 2048x2048
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)
- **Alpha hints**: HSV chroma key (auto-generated from green screen footage)
- **Color pipeline**: `input_is_linear=True` — engine handles linear->sRGB conversion internally
- **Profiles benchmarked**: v2

> **Note**: The original engine OOMs at 4K on 8 GB GPUs. Flash Attention is the
> minimum required optimization. The "baseline" config uses Flash Attention only
> to serve as the closest proxy to original behavior while remaining runnable.

---

## Results

### V2 (optimized + torch.compile)

| Metric | Value |
|---|---:|
| Total time | 646.5 s |
| Effective FPS | 0.15 fps |
| Avg frame time | 3946.0 ms |
| Median frame time | 3563.1 ms |
| Min frame time | 3198.4 ms |
| Max frame time | 44430.4 ms |

#### GPU Memory

| Metric | Value |
|---|---:|
| Dedicated VRAM (physical) | 4915.5 MB |
| PyTorch allocator reserved | 3794.0 MB |
| Idle after model load | 1905.5 MB |
| Model load time | 2550 ms |

> Fits within **8188 MB** dedicated VRAM with **4394 MB headroom**.

#### Warmup (first 5 avg: 11735 ms, last 5 avg: 3456 ms, overhead: +239.6%)

---

## Output Files

### V2 (optimized + torch.compile)

- **Composite**: `Output\comp_v2`
- **Alpha matte**: `Output\alpha_v2`

> Output is linear premultiplied RGBA EXR — ready for compositing in Nuke/Fusion/etc.

---

## Configuration Details

### V2 (optimized + torch.compile)

- **Config**: OptimizationConfig: flash_attention, tiled_refiner(512x512/128px), sparse_refiner, disable_cudnn_benchmark, compile_submodules
- **Active optimizations**: flash_attention, tiled_refiner(512x512/128px), sparse_refiner, disable_cudnn_benchmark, compile_submodules
- **Model load time**: 2550 ms
- **Frames processed**: 100
- **Wall time (incl. subprocess)**: 655.8 s
