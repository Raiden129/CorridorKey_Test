"""4K Benchmark: Configurable profile benchmarking for CorridorKey.

Processes real 4K green-screen footage (Tears of Steel, 4096x2160 EXR frames
in linear color space) through a selected optimization profile.

Available profiles:
  - baseline    Flash Attention only (minimum required to avoid OOM on 8 GB)
  - optimized   Flash + tiled refiner + cache clearing + cuDNN disable + sparse refiner
  - v2          All optimized + torch.compile sub-modules, cache clearing disabled
  - all         Run baseline + optimized + v2 and generate comparison report

Each config runs in its own subprocess.  Outputs comp and alpha EXR sequences
for quality inspection.

Source footage:
  Tears of Steel (CC-BY 3.0) (c) Blender Foundation | mango.blender.org
  https://media.xiph.org/tearsofsteel/tearsofsteel-footage-exr/02_3c/linear/

Usage:
    uv run python benchmark_4k_vram.py                   # defaults to v2
    uv run python benchmark_4k_vram.py --profile v2
    uv run python benchmark_4k_vram.py --profile optimized
    uv run python benchmark_4k_vram.py --profile baseline
    uv run python benchmark_4k_vram.py --profile all     # compare all profiles
    uv run python benchmark_4k_vram.py --profile baseline v2  # compare two
    uv run python benchmark_4k_vram.py --frames 50       # fewer frames
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CKPT = os.path.join("CorridorKeyModule", "checkpoints", "CorridorKey.pth")
FRAMES_DIR = os.path.join("tears_of_steel_test", "frames")
HINT_DIR = os.path.join("tears_of_steel_test", "alpha_hints")
OUTPUT_DIR = "Output"
REPORT_PATH = "benchmark_4k_results.md"

PROFILE_CONFIGS: dict[str, tuple[str, dict]] = {
    "baseline": ("Flash Attention Only (baseline)", {
        "flash_attention": True,
    }),
    "optimized": ("All Optimizations", {
        "flash_attention": True,
        "tiled_refiner": True,
        "disable_cudnn_benchmark": True,
        "cache_clearing": True,
        "sparse_refiner": True,
    }),
    "v2": ("V2 (optimized + torch.compile)", {
        "flash_attention": True,
        "tiled_refiner": True,
        "disable_cudnn_benchmark": True,
        "cache_clearing": False,
        "sparse_refiner": True,
        "compile_submodules": True,
    }),
}

# ---------------------------------------------------------------------------
# Worker script (runs in subprocess)
# ---------------------------------------------------------------------------

WORKER_SCRIPT = r'''
"""Worker: process 4K EXR frame sequence, write output EXRs, report metrics."""
import json, sys, time, os, glob, contextlib, io, threading

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# GPU memory poller -- samples actual device memory
# ---------------------------------------------------------------------------
class GPUMemoryPoller:
    """Background thread that polls torch.cuda.mem_get_info() to track
    the real peak GPU memory usage, including cuDNN workspace and other
    allocations outside PyTorch's caching allocator."""

    def __init__(self, device=0, interval_ms=50):
        self.device = device
        self.interval = interval_ms / 1000.0
        self._peak_used_bytes = 0
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        # Get total GPU memory once
        _, self.total_bytes = torch.cuda.mem_get_info(self.device)

    def _poll_loop(self):
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info(self.device)
            used = total - free
            if used > self._peak_used_bytes:
                self._peak_used_bytes = used
            self._samples.append(used)
            self._stop.wait(self.interval)

    def start(self):
        self._peak_used_bytes = 0
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def reset_peak(self):
        """Reset peak for per-frame tracking."""
        self._peak_used_bytes = 0

    @property
    def peak_used_mb(self):
        return self._peak_used_bytes / (1024**2)

    @property
    def current_used_mb(self):
        free, total = torch.cuda.mem_get_info(self.device)
        return (total - free) / (1024**2)

    @property
    def total_mb(self):
        return self.total_bytes / (1024**2)

config_json  = sys.argv[1]
ckpt_path    = sys.argv[2]
frames_dir   = sys.argv[3]
output_dir   = sys.argv[4]
tag          = sys.argv[5]
hint_dir     = sys.argv[6]
num_frames   = int(sys.argv[7])

config_fields = json.loads(config_json)

# ---- Build config ----
from CorridorKeyModule.optimization_config import OptimizationConfig
config = OptimizationConfig(enable_metrics=True, **config_fields)

if config_fields:
    from CorridorKeyModule.optimized_engine import OptimizedCorridorKeyEngine as EngineClass
else:
    from CorridorKeyModule.inference_engine import CorridorKeyEngine as EngineClass

# ---- Discover EXR frames and alpha hints ----
exr_files = sorted(glob.glob(os.path.join(frames_dir, "*.exr")))[:num_frames]
hint_files = sorted([f for f in os.listdir(hint_dir) if f.lower().endswith(('.png', '.exr', '.jpg', '.tif', '.tiff'))]) if os.path.isdir(hint_dir) else []
total_frames = len(exr_files)

if total_frames == 0:
    print("ERROR: No EXR frames found", flush=True)
    sys.exit(1)

# Read first frame to get dimensions
first_frame = cv2.imread(exr_files[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
h, w = first_frame.shape[:2]
fps = 24.0  # Tears of Steel footage is 24 fps

print(f"Found {total_frames} EXR frames ({w}x{h}) and {len(hint_files)} alpha hints", flush=True)

result = {
    "resolution": f"{w}x{h}",
    "total_frames": total_frames,
    "fps": fps,
    "config_summary": config.summary(),
    "active_opts": config.active_optimizations(),
    "tag": tag,
}

# ---- Load engine ----
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

load_start = time.perf_counter()
engine = EngineClass(
    checkpoint_path=ckpt_path,
    device="cuda",
    img_size=2048,
    use_refiner=True,
    optimization_config=config,
)
load_time_ms = (time.perf_counter() - load_start) * 1000
model_vram_mb = torch.cuda.memory_allocated() / (1024**2)

result["model_vram_mb"] = round(model_vram_mb, 1)
result["load_time_ms"] = round(load_time_ms, 1)

# ---- Start GPU memory poller ----
poller = GPUMemoryPoller(device=0, interval_ms=25)
idle_gpu_mb = poller.current_used_mb
result["idle_gpu_mb"] = round(idle_gpu_mb, 1)
print(f"Device memory after model load: {idle_gpu_mb:.0f} MB (this is the idle baseline)", flush=True)
poller.start()

# ---- Prepare output directories ----
comp_dir  = os.path.join(output_dir, f"comp_{tag}")
alpha_dir = os.path.join(output_dir, f"alpha_{tag}")
os.makedirs(comp_dir, exist_ok=True)
os.makedirs(alpha_dir, exist_ok=True)

# EXR write params: PXR24 compression, half-float
exr_params = [
    cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]

# ---- Process frames ----
torch.cuda.reset_peak_memory_stats()

frame_times = []
inference_stage_times = []
postprocess_stage_times = []
vram_peaks_per_frame = []
device_peaks_per_frame = []
overall_start = time.perf_counter()

for frame_idx, exr_path in enumerate(exr_files):
    # Read EXR frame (linear float, BGR)
    frame_bgr = cv2.imread(exr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if frame_bgr is None:
        print(f"  WARNING: Failed to read {exr_path}, skipping", flush=True)
        continue

    # Convert BGR to RGB, ensure float32
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Load alpha hint for this frame
    if frame_idx < len(hint_files):
        hint_path = os.path.join(hint_dir, hint_files[frame_idx])
        mask_raw = cv2.imread(hint_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            print(f"  WARNING: Failed to read hint {hint_path}, using zeros", flush=True)
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            # Handle multi-channel (take first channel)
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[:, :, 0]
            # Normalize to 0-1
            if mask_raw.dtype == np.uint8:
                mask = mask_raw.astype(np.float32) / 255.0
            elif mask_raw.dtype == np.uint16:
                mask = mask_raw.astype(np.float32) / 65535.0
            else:
                mask = mask_raw.astype(np.float32)
            # Resize to match frame dimensions if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        print(f"  WARNING: No hint for frame {frame_idx}, using zeros", flush=True)
        mask = np.zeros((h, w), dtype=np.float32)

    torch.cuda.reset_peak_memory_stats()
    poller.reset_peak()
    t0 = time.perf_counter()

    # Suppress per-frame print output from engine
    with contextlib.redirect_stdout(io.StringIO()):
        output = engine.process_frame(frame_rgb, mask, input_is_linear=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    device_peak_mb = poller.peak_used_mb

    frame_times.append(elapsed_ms)
    vram_peaks_per_frame.append(peak_mb)
    device_peaks_per_frame.append(device_peak_mb)

    if "metrics" in output:
        for s in output["metrics"].stages:
            if s.name == "inference":
                inference_stage_times.append(s.duration_ms)
            elif s.name == "postprocess":
                postprocess_stage_times.append(s.duration_ms)

    # Write processed RGBA EXR (linear premultiplied - ready for compositing)
    processed = output["processed"]  # [H, W, 4] linear float RGBA
    # Convert RGBA to BGRA for OpenCV
    processed_bgra = cv2.cvtColor(processed, cv2.COLOR_RGBA2BGRA)
    basename = os.path.splitext(os.path.basename(exr_path))[0]
    cv2.imwrite(os.path.join(comp_dir, f"{basename}.exr"), processed_bgra, exr_params)

    # Write alpha channel as single-channel EXR
    alpha = output["alpha"]
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    cv2.imwrite(os.path.join(alpha_dir, f"{basename}.exr"), alpha, exr_params)

    frame_idx_display = frame_idx + 1
    # Progress every 10 frames
    if frame_idx_display % 10 == 0 or frame_idx_display == total_frames:
        avg_ms = sum(frame_times) / len(frame_times)
        eta_s = (total_frames - frame_idx_display) * avg_ms / 1000
        print(f"  [{frame_idx_display}/{total_frames}] "
              f"last={elapsed_ms:.0f}ms avg={avg_ms:.0f}ms "
              f"device_peak={device_peak_mb:.0f}MB ETA={eta_s:.0f}s",
              flush=True)

overall_elapsed_s = time.perf_counter() - overall_start
poller.stop()

# ---- Collect final metrics ----
overall_peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
overall_peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)

ft = frame_times
result["status"] = "OK"
result["overall_time_s"] = round(overall_elapsed_s, 2)
result["frames_processed"] = len(frame_times)
result["avg_frame_ms"] = round(sum(ft) / len(ft), 1)
result["min_frame_ms"] = round(min(ft), 1)
result["max_frame_ms"] = round(max(ft), 1)
result["median_frame_ms"] = round(sorted(ft)[len(ft)//2], 1)
result["effective_fps"] = round(len(frame_times) / overall_elapsed_s, 2)
result["peak_vram_allocated_mb"] = round(overall_peak_alloc, 1)
result["peak_vram_reserved_mb"] = round(overall_peak_reserved, 1)
result["avg_vram_peak_per_frame_mb"] = round(sum(vram_peaks_per_frame) / len(vram_peaks_per_frame), 1)
result["max_vram_peak_per_frame_mb"] = round(max(vram_peaks_per_frame), 1)

# Device-level GPU memory (actual usage including cuDNN, CUDA context, etc.)
result["device_peak_gpu_mb"] = round(max(device_peaks_per_frame), 1)
result["device_avg_peak_gpu_mb"] = round(sum(device_peaks_per_frame) / len(device_peaks_per_frame), 1)
result["device_idle_gpu_mb"] = round(idle_gpu_mb, 1)
result["device_total_gpu_mb"] = round(poller.total_mb, 1)

result["comp_output"] = comp_dir
result["alpha_output"] = alpha_dir

if inference_stage_times:
    result["avg_inference_ms"] = round(sum(inference_stage_times) / len(inference_stage_times), 1)
if postprocess_stage_times:
    result["avg_postprocess_ms"] = round(sum(postprocess_stage_times) / len(postprocess_stage_times), 1)

# First 5 frame times (warmup check) vs last 5
if len(ft) >= 10:
    result["first5_avg_ms"] = round(sum(ft[:5]) / 5, 1)
    result["last5_avg_ms"] = round(sum(ft[-5:]) / 5, 1)

print("===RESULT_JSON===")
print(json.dumps(result))
'''

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_config(label: str, config_fields: dict, tag: str, num_frames: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {label}")
    print(f"  Processing {num_frames} 4K EXR frames (Tears of Steel)")
    print(f"{'='*60}")

    config_json = json.dumps(config_fields)
    cmd = [
        sys.executable, "-c", WORKER_SCRIPT,
        config_json, CKPT, FRAMES_DIR, OUTPUT_DIR, tag, HINT_DIR, str(num_frames),
    ]

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall_time = time.time() - start

    if proc.stdout:
        for line in proc.stdout.splitlines():
            if not line.startswith("===RESULT_JSON==="):
                print(line)

    if proc.stderr:
        for line in proc.stderr.splitlines():
            if line.strip():
                print(f"  [stderr] {line}")

    result = {"label": label, "wall_time_s": round(wall_time, 1)}

    if "===RESULT_JSON===" in proc.stdout:
        json_str = proc.stdout.split("===RESULT_JSON===\n", 1)[1].strip()
        try:
            result.update(json.loads(json_str))
        except json.JSONDecodeError:
            result["status"] = "PARSE_ERROR"
            result["error"] = json_str[:200]
    else:
        result["status"] = "CRASH"
        result["error"] = (proc.stderr or proc.stdout or "No output")[:300]

    status = result.get("status", "UNKNOWN")
    if status == "OK":
        fps = result.get("effective_fps", 0)
        device_peak = result.get("device_peak_gpu_mb", 0)
        total = result.get("overall_time_s", 0)
        print(f"\n  DONE: {result.get('frames_processed',0)} frames in {total:.1f}s "
              f"({fps:.2f} fps) | Device peak: {device_peak:.0f} MB")
    else:
        print(f"\n  {status}: {result.get('error', 'unknown')[:100]}")

    return result


def generate_report(results: list[dict], gpu_info: str, num_frames: int) -> str:
    L: list[str] = []
    ok_results = [r for r in results if r.get("status") == "OK"]
    is_comparison = len(ok_results) > 1

    L.append("# CorridorKey 4K Benchmark — Tears of Steel")
    L.append("")
    L.append("## Test Configuration")
    L.append("")
    L.append("- **Source footage**: Tears of Steel (scene 02_3c) — CC-BY 3.0 (c) Blender Foundation | mango.blender.org")
    L.append("- **Format**: OpenEXR 16-bit half-float, linear color space")
    L.append("- **Resolution**: 4096x2160 (DCI 4K)")
    L.append(f"- **Frames**: {num_frames} (24 fps, ~{num_frames/24:.1f} seconds)")
    L.append("- **Model input size**: 2048x2048")
    L.append(f"- **GPU**: {gpu_info}")
    L.append("- **Alpha hints**: HSV chroma key (auto-generated from green screen footage)")
    L.append("- **Color pipeline**: `input_is_linear=True` — engine handles linear->sRGB conversion internally")
    L.append(f"- **Profiles benchmarked**: {', '.join(r.get('tag', '?') for r in ok_results)}")
    L.append("")
    L.append("> **Note**: The original engine OOMs at 4K on 8 GB GPUs. Flash Attention is the")
    L.append("> minimum required optimization. The \"baseline\" config uses Flash Attention only")
    L.append("> to serve as the closest proxy to original behavior while remaining runnable.")
    L.append("")

    # --- Comparison table (multiple profiles) ---
    if is_comparison:
        L.append("---")
        L.append("")
        L.append("## Comparison")
        L.append("")

        # Build header
        header_cols = ["Metric"] + [r["label"] for r in ok_results]
        if len(ok_results) == 2:
            header_cols.append("Delta")
        L.append("| " + " | ".join(header_cols) + " |")
        L.append("|---" + "|---:" * (len(header_cols) - 1) + "|")

        def comp_row(label: str, key: str, unit: str = "", fmt: str = ".1f"):
            vals = [r.get(key, 0) for r in ok_results]
            cols = [label] + [f"{v:{fmt}} {unit}" for v in vals]
            if len(ok_results) == 2 and all(isinstance(v, (int, float)) for v in vals):
                delta = vals[1] - vals[0]
                if vals[0] != 0:
                    pct = (delta / vals[0]) * 100
                    sign = "+" if delta > 0 else ""
                    cols.append(f"{sign}{delta:{fmt}} {unit} ({sign}{pct:.1f}%)")
                else:
                    cols.append(f"{delta:{fmt}} {unit}")
            L.append("| " + " | ".join(cols) + " |")

        comp_row("Total time", "overall_time_s", "s")
        comp_row("Effective FPS", "effective_fps", "fps", fmt=".2f")
        comp_row("Avg frame time", "avg_frame_ms", "ms")
        comp_row("Median frame time", "median_frame_ms", "ms")
        comp_row("Min frame time", "min_frame_ms", "ms")
        comp_row("Max frame time", "max_frame_ms", "ms")

        L.append("")
        L.append("#### GPU Memory")
        L.append("")

        header_cols = ["Metric"] + [r["label"] for r in ok_results]
        if len(ok_results) == 2:
            header_cols.append("Delta")
        L.append("| " + " | ".join(header_cols) + " |")
        L.append("|---" + "|---:" * (len(header_cols) - 1) + "|")

        comp_row("Dedicated VRAM (physical)", "device_peak_gpu_mb", "MB")
        comp_row("PyTorch allocator reserved", "peak_vram_reserved_mb", "MB")
        comp_row("Idle after model load", "device_idle_gpu_mb", "MB")
        L.append("")

        # Warmup analysis
        if all("first5_avg_ms" in r for r in ok_results):
            L.append("### Warmup Effect (first 5 vs last 5 frames)")
            L.append("")
            L.append("| Profile | First 5 avg (ms) | Last 5 avg (ms) | Warmup overhead |")
            L.append("|---|---:|---:|---:|")
            for r in ok_results:
                f5, l5 = r["first5_avg_ms"], r["last5_avg_ms"]
                wo = ((f5 - l5) / l5 * 100) if l5 else 0
                L.append(f"| {r['label']} | {f5:.0f} | {l5:.0f} | {wo:+.1f}% |")
            L.append("")

    # --- Single profile results ---
    else:
        L.append("---")
        L.append("")
        L.append("## Results")
        L.append("")

        for r in ok_results:
            L.append(f"### {r['label']}")
            L.append("")
            L.append("| Metric | Value |")
            L.append("|---|---:|")
            L.append(f"| Total time | {r.get('overall_time_s', 0):.1f} s |")
            L.append(f"| Effective FPS | {r.get('effective_fps', 0):.2f} fps |")
            L.append(f"| Avg frame time | {r.get('avg_frame_ms', 0):.1f} ms |")
            L.append(f"| Median frame time | {r.get('median_frame_ms', 0):.1f} ms |")
            L.append(f"| Min frame time | {r.get('min_frame_ms', 0):.1f} ms |")
            L.append(f"| Max frame time | {r.get('max_frame_ms', 0):.1f} ms |")
            L.append("")
            L.append("#### GPU Memory")
            L.append("")
            L.append("| Metric | Value |")
            L.append("|---|---:|")
            L.append(f"| Dedicated VRAM (physical) | {r.get('device_peak_gpu_mb', 0):.1f} MB |")
            L.append(f"| PyTorch allocator reserved | {r.get('peak_vram_reserved_mb', 0):.1f} MB |")
            L.append(f"| Idle after model load | {r.get('device_idle_gpu_mb', 0):.1f} MB |")
            L.append(f"| Model load time | {r.get('load_time_ms', 0):.0f} ms |")

            total_gpu = r.get("device_total_gpu_mb", 0)
            reserved = r.get("peak_vram_reserved_mb", 0)
            spillover = max(0, reserved - total_gpu)
            L.append("")
            if spillover > 0:
                L.append(f"> **Shared GPU memory spillover**: PyTorch reserved **{reserved:.0f} MB** but the GPU")
                L.append(f"> only has **{total_gpu:.0f} MB** dedicated VRAM — **~{spillover:.0f} MB spilled into")
                L.append(f"> shared GPU memory** (system RAM over PCIe).")
            else:
                headroom = total_gpu - reserved
                L.append(f"> Fits within **{total_gpu:.0f} MB** dedicated VRAM with **{headroom:.0f} MB headroom**.")
            L.append("")

            if "first5_avg_ms" in r:
                f5, l5 = r["first5_avg_ms"], r["last5_avg_ms"]
                wo = ((f5 - l5) / l5 * 100) if l5 else 0
                L.append(f"#### Warmup (first 5 avg: {f5:.0f} ms, last 5 avg: {l5:.0f} ms, overhead: {wo:+.1f}%)")
                L.append("")

    # --- Output files ---
    L.append("---")
    L.append("")
    L.append("## Output Files")
    L.append("")
    for r in ok_results:
        L.append(f"### {r['label']}")
        L.append("")
        L.append(f"- **Composite**: `{r.get('comp_output', 'N/A')}`")
        L.append(f"- **Alpha matte**: `{r.get('alpha_output', 'N/A')}`")
        L.append("")

    L.append("> Output is linear premultiplied RGBA EXR — ready for compositing in Nuke/Fusion/etc.")
    L.append("")

    # --- Configuration details ---
    L.append("---")
    L.append("")
    L.append("## Configuration Details")
    L.append("")
    for r in ok_results:
        L.append(f"### {r['label']}")
        L.append("")
        L.append(f"- **Config**: {r.get('config_summary', 'N/A')}")
        L.append(f"- **Active optimizations**: {', '.join(r.get('active_opts', [])) or 'none'}")
        L.append(f"- **Model load time**: {r.get('load_time_ms', 0):.0f} ms")
        L.append(f"- **Frames processed**: {r.get('frames_processed', 0)}")
        L.append(f"- **Wall time (incl. subprocess)**: {r.get('wall_time_s', 0):.1f} s")
        L.append("")

    return "\n".join(L)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CorridorKey 4K Benchmark — Tears of Steel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        nargs="+",
        choices=["baseline", "optimized", "v2", "all"],
        default=["v2"],
        help="Profile(s) to benchmark. Use 'all' to run baseline + optimized + v2. "
             "Specify multiple to compare (e.g. --profile baseline v2). Default: v2",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory for EXR sequences (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--report",
        default=REPORT_PATH,
        help=f"Path for the markdown report (default: {REPORT_PATH})",
    )
    return parser.parse_args()


def main():
    import shutil

    args = parse_args()
    num_frames = args.frames
    output_dir = args.output_dir
    report_path = args.report

    # Expand "all" into individual profiles
    profiles: list[str] = []
    for p in args.profile:
        if p == "all":
            profiles.extend(["baseline", "optimized", "v2"])
        elif p not in profiles:
            profiles.append(p)

    # Update globals used by run_config
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared {output_dir}/")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(CKPT):
        print(f"ERROR: Checkpoint not found: {CKPT}")
        sys.exit(1)
    if not os.path.isdir(FRAMES_DIR):
        print(f"ERROR: Frames directory not found: {FRAMES_DIR}")
        print("Run: uv run python tears_of_steel_test/download_frames.py")
        sys.exit(1)

    import glob as _glob
    available_frames = len(_glob.glob(os.path.join(FRAMES_DIR, "*.exr")))
    if available_frames < num_frames:
        print(f"WARNING: Only {available_frames} EXR frames available (need {num_frames})")
        print("Run: uv run python tears_of_steel_test/download_frames.py")
    if not os.path.isdir(HINT_DIR):
        print(f"ERROR: Alpha hint directory not found: {HINT_DIR}")
        print("Run: uv run python tears_of_steel_test/generate_alpha_hints.py")
        sys.exit(1)
    hint_count = len([f for f in os.listdir(HINT_DIR) if f.lower().endswith(('.png', '.exr', '.jpg'))])
    if hint_count == 0:
        print(f"ERROR: No alpha hints found in {HINT_DIR}")
        print("Run: uv run python tears_of_steel_test/generate_alpha_hints.py")
        sys.exit(1)

    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info = f"{gpu_name} ({gpu_mem:.1f} GB)"
    except Exception:
        gpu_info = "Unknown"

    print("=" * 60)
    print("  CORRIDORKEY 4K BENCHMARK — TEARS OF STEEL")
    print(f"  GPU: {gpu_info}")
    print(f"  Frames: {available_frames} EXR frames ({FRAMES_DIR})")
    print(f"  Alpha hints: {hint_count} ({HINT_DIR})")
    print(f"  Profiles: {', '.join(profiles)}")
    print(f"  Frame count: {num_frames}")
    print("=" * 60)

    results = []
    for profile_name in profiles:
        label, config_fields = PROFILE_CONFIGS[profile_name]
        result = run_config(label, config_fields, profile_name, num_frames)
        result["tag"] = profile_name
        results.append(result)

    report = generate_report(results, gpu_info, num_frames)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Report: {report_path}")
    print(f"{'='*60}\n")

    for r in results:
        if r.get("status") == "OK":
            print(f"  {r['label']}:")
            print(f"    {r.get('effective_fps',0):.2f} fps | "
                  f"{r.get('overall_time_s',0):.1f}s total | "
                  f"Device peak: {r.get('device_peak_gpu_mb',0):.0f} MB | "
                  f"Allocator reserved: {r.get('peak_vram_reserved_mb',0):.0f} MB")
            print(f"    Outputs: {r.get('comp_output','')}, {r.get('alpha_output','')}")


if __name__ == "__main__":
    main()
