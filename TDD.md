# Technical Design Document (TDD): CorridorKey (Reverse-Engineered from `Raiden129/CorridorKey_Test`)

## [SECTION: END-TO-END PIPELINE FLOW TRACE]

### PASS 1 — THE NARRATIVE TRACE

A single frame enters CorridorKey through `clip_manager.run_inference()` (or `backend.service.CorridorKeyService.run_inference()` in the service path). In the CLI path, frame arrival is synchronous and pull-based.

**Stage 0 — Frame Arrival**

For input **video** clips, OpenCV `cv2.VideoCapture.read()` decodes one frame at a time on CPU into a BGR `uint8` array shaped `[H, W, 3]`. The code immediately converts BGR→RGB (`cv2.cvtColor`) and scales to float32 `[0,1]`.

For input **image sequences**, each file is read from disk:
- EXR: `cv2.imread(..., IMREAD_UNCHANGED)` returns float data, possible BGRA; alpha is dropped if present, then BGR→RGB. Negative values are clamped to 0. If requested, linear→sRGB conversion is applied.
- Non-EXR: `cv2.imread` returns BGR `uint8`, converted to RGB float32 `[0,1]`.

Memory is CPU-owned NumPy arrays (row-major C-order [INFERRED], no explicit stride alignment control). No memory-mapped or zero-copy decode path exists. Ownership stays with Python local variables until handed to the engine; then tensors are created and old arrays are dereferenced by Python GC.

**Stage 1 — Frame Pre-Processing**

`CorridorKeyEngine.process_frame()` accepts `image` as `[H,W,3]` or `[B,H,W,3]`.

1. Adds batch dimension if needed.
2. Converts NumPy→Torch with `torch.from_numpy(...).permute(0,3,1,2)`, producing BCHW.
3. `TF.to_dtype(..., scale=True)` converts to model dtype (`float16` by default via backend) and normalizes integer input to `[0,1]`; float inputs are preserved in numeric range [INFERRED from torchvision semantics].
4. Transfers to target device (`.to(device, non_blocking=True)`).
5. Resizes image to fixed `img_size` (default `2048x2048`) using bilinear interpolation.
6. If `input_is_linear=True`, converts linear→sRGB after resize.
7. Applies ImageNet normalization with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
8. Concatenates mask channel (later) to form 4-channel input.

Aspect ratio is **not preserved**; frame is warped to square. No letterboxing/cropping/tiling in Torch backend inference path. Preprocessing runs on whichever device tensor is on (CPU or GPU). Operations are torchvision/PyTorch kernels, not custom CUDA kernels.

**Stage 2 — Alpha Hint Mask Ingestion & Encoding**

Mask source:
- Video alpha hint: blue channel extracted from BGR frame (`frame[:,:,2]`) and scaled to `[0,1]`.
- Image alpha hint: read with depth-preserving flag; first channel taken if multi-channel; dtype normalized (`uint8/255`, `uint16/65535`, else cast float32).
- If mask resolution differs from input frame, resized with linear interpolation to match frame size.

Mask format at engine boundary is dense float mask `[H,W]` (or batch equivalent). No sparse point/brush encoding, no trimap semantic encoding, no morphological preconditioning before model. It is encoded as a **4th input channel** by channel concatenation:
- Final model input tensor: `float{16|32} [B,4,2048,2048]`.

**Stage 3 — GreenFormer Inference**

`GreenFormer` pipeline:

1. **Input projection / patch embedding**
   - Uses TIMM Hiera backbone (`hiera_base_plus_224...`) with patched first patch-embedding conv to accept 4 channels (RGB+hint). Extra channel weights initialized to zero then learned from checkpoint.
2. **Backbone feature extraction**
   - Hiera encoder returns 4 scales (channels typically `[112,224,448,896]`).
   - Attention/token mechanics are inside TIMM Hiera implementation (global/local strategy not redefined in CorridorKey code). Hint is not separately attended; it is only present as channel 4 at input.
3. **Decoder heads**
   - Two parallel multiscale decoder heads (`DecoderHead`): one for alpha logits (1ch), one for FG logits (3ch).
   - Each head linearly embeds each scale, upsamples to `H/4`, fuses, predicts logits.
4. **Upsampling**
   - Decoder logits are bilinearly upsampled to full input resolution.
5. **Refiner**
   - Coarse probabilities are `sigmoid(logits)`.
   - A CNN refiner gets `[RGB + coarse_alpha + coarse_fg]` (7 channels), produces 4 delta logits.
   - Deltas are added in logit space to coarse logits (residual correction).
6. **Final activations**
   - Final alpha and FG are sigmoid outputs in `[0,1]`.

Output tensor shapes:
- Alpha: `[B,1,H,W]`
- FG: `[B,3,H,W]`
where `H,W` are model input size (`2048` default).

Inference uses `torch.autocast` float16 when enabled.

**Stage 4 — Alpha Post-Processing**

After model output, both alpha and fg are resized back to original frame resolution:
- GPU path: torchvision bilinear resize.
- CPU path: OpenCV Lanczos4 resize.

Raw alpha returned as `res["alpha"]` is **not thresholded** and not clamped explicitly post-resize (it remains approximately `[0,1]` due to sigmoid + interpolation).  
Optional matte cleanup (`auto_despeckle`) is applied to `processed_alpha`, not to the raw `alpha` output:
- CPU: connected components on thresholded alpha (`>0.5`), remove small components by area, then dilation (default radius 25), Gaussian blur (size 5), multiply back with original alpha.
- GPU: tensor connected-component approximation, threshold `>0.25`, keep components above area threshold, iterative dilation and gaussian blur.

No guided filtering. No temporal smoothing. No feedback from previous frames.

If resizing occurred, alpha returns to original input resolution at this stage.

**Stage 5 — Green Spill Suppression**

Spill suppression happens **after model inference** and after optional matte cleanup, during postprocessing:
- Applied to foreground RGB only.
- Targeted on all pixels where `G > limit`, with `limit=(R+B)/2` (default “average” mode); i.e., not explicitly alpha-gated.
- Spill amount = `max(G-limit, 0)`.
- New channels:
  - `G' = G - spill`
  - `R' = R + 0.5*spill`
  - `B' = B + 0.5*spill`
- Blended by `strength` (0..1).

This is done in sRGB-encoded space first, then converted to linear for EXR premultiplication/composite. Luminance preservation is approximate via channel redistribution; no explicit luminance-constraint solve.

**Stage 6 — Foreground Reconstruction**

No algebraic solve of `C = αF + (1−α)B` is performed in inference code. Foreground `F` is directly predicted by GreenFormer (`res["fg"]`, straight sRGB).  
Background green estimation is bypassed. No denominator instability branch (`α≈0`) exists because no division-based reconstruction path is used.

For `processed` output, FG is converted to linear and premultiplied by cleaned alpha:
- `F_linear = srgb_to_linear(F_despilled)`
- `F_premul = F_linear * α_processed`
- `processed = concat(F_premul, α_processed)` as RGBA float.

**Stage 7 — Compositing**

Internal compositing is preview-only unless caller does external comp:
- Background = generated checkerboard (`color1=0.15`, `color2=0.55` in sRGB), then converted to linear.
- Operation (straight path): `Out_linear = α*F_linear + (1−α)*BG_linear`.
- Output converted back to sRGB for preview (`comp`).

Runs on GPU or CPU depending on post-processing path.  
Output `comp` format in memory: float32 RGB `[H,W,3]` in `[0,1]`.

No LUT/color grading stage is applied.

**Stage 8 — Output & Write**

`clip_manager` writes four products per frame:
- `FG/*.exr` : raw predicted straight FG in sRGB gamut (RGB float32, written via OpenCV EXR half-float PXR24).
- `Matte/*.exr` : raw alpha (single-channel linear float).
- `Processed/*.exr` : linear premultiplied RGBA float.
- `Comp/*.png` : checkerboard preview in 8-bit sRGB.

Write path uses per-frame allocations; no explicit output buffer pool. Handoff is filesystem writes via `cv2.imwrite` (and optional FFmpeg stitch for comp video).  
Float→uint8 quantization for comp is `(clip(x,0,1)*255).astype(np.uint8)` (truncate toward zero, no dithering).  
Timestamps are not propagated in image sequence outputs; frame index is preserved in filename stem. For video comp stitching, source FPS is probed and reused.

---

### PASS 2 — THE ANNOTATED DATA FLOW TABLE

| Stage | Input Format | Operation | Output Format | Where Executed | Potential Failure / Edge Case |
|---|---|---|---|---|---|
| 0 Frame Arrival | Video: BGR `uint8 [H,W,3]`; Image EXR: BGR/BGRA float; Image std: BGR `uint8` | Decode/read + BGR→RGB + (for std images) normalize to float | RGB `float32 [H,W,3]` | CPU (OpenCV) | Missing/corrupt frame; EXR alpha channel unexpectedly present; video read failure mid-stream |
| 1 Preprocess | RGB `float32/uint8 [B,H,W,3]` | NumPy→Torch BCHW, dtype conversion, resize to fixed square, optional linear→sRGB, ImageNet normalize | `float{16|32} [B,3,S,S]` (`S=img_size`) | CPU or GPU (Torch/torchvision) | Aspect-ratio distortion; OOM on large `S`; wrong `input_is_linear` causing gamma errors |
| 2 Hint Ingestion | Mask from video channel or image file (`uint8/uint16/float`) | Channel normalize, dtype normalize to `[0,1]`, resize to frame size, reshape to `[B,1,H,W]`, concat as 4th channel | `float{16|32} [B,4,S,S]` | CPU for I/O + CPU/GPU for tensor ops | Mask missing; malformed channel count; size mismatch; wrong mask range |
| 3 GreenFormer Inference | `float{16|32} [B,4,S,S]` | Hiera encode + dual decoders + upsample + optional CNN refiner delta logits + sigmoid | Alpha `float [B,1,S,S]`, FG `float [B,3,S,S]` | GPU preferred, else CPU (PyTorch) | Missing checkpoint; backend incompatibility; OOM; compile failure fallback to eager |
| 4 Alpha Postprocess | Alpha `float [B,1,S,S]` | Resize to original `H,W`; optional despeckle (CC+dilate+blur) | Raw alpha `float32 [H,W,1]`; cleaned alpha* `float32 [H,W,1]` | GPU or CPU depending path | Over-aggressive cleanup erodes details; no temporal stabilizer → flicker |
| 5 Spill Suppression | FG raw sRGB `float [H,W,3]` | Luminance-ish despill with strength blending | FG despilled sRGB `float32 [H,W,3]` | GPU (`despill_torch`) or CPU (`despill_opencv`) | Skin/wardrobe color shifts; applied globally (not alpha-gated) |
| 6 Foreground Reconstruction | FG despilled sRGB + cleaned alpha | sRGB→linear, premultiply, pack RGBA | Processed `float32 [H,W,4]` linear premul | GPU or CPU | If alpha noisy, premul embeds matte defects |
| 7 Compositing | FG linear + alpha + checkerboard BG (linear) | `Out=αF+(1-α)B`, linear→sRGB | Comp preview `float32 [H,W,3]` sRGB | GPU or CPU | Only checkerboard bg supported in this internal path; no temporal consistency |
| 8 Output Write | FG/alpha/processed float arrays; comp float | Color channel reorders, EXR half-float write, PNG quantize/write, optional FFmpeg stitch | Disk files: EXR/PNG/MP4 | CPU (OpenCV/FFmpeg) | Disk IO failure; ffmpeg missing; quantization banding in PNG |

\*inferred/derived output only used for processed/comp, not the raw `Matte` export.

---

### PASS 3 — THE DECISION MAP

1. **Decision:** Input source mode  
   **Condition:** `clip.input_asset.type == "video"`  
   **Branch A:** VideoCapture decode sequential frames; stem = zero-padded index.  
   **Branch B:** Image sequence read by filename; stem from file basename.  
   **Risk:** Wrong branch causes frame indexing/stem mismatch and resume/write confusion.

2. **Decision:** Alpha source mode  
   **Condition:** `clip.alpha_asset.type == "video"`  
   **Branch A:** Use blue channel from decoded frame.  
   **Branch B:** Read image/depth mask file and normalize dtype/channels.  
   **Risk:** Silent channel misuse can invert/flatten mask fidelity.

3. **Decision:** EXR gamma handling  
   **Condition:** input file `.exr` and `input_is_linear` flag  
   **Branch A (`input_is_linear=True`):** keep linear data path.  
   **Branch B (`False`):** gamma-correct EXR for sRGB model path.  
   **Risk:** Wrong branch yields dark fringes / energy mismatch.

4. **Decision:** Mask/frame resolution mismatch  
   **Condition:** `mask.shape[:2] != img.shape[:2]`  
   **Branch A:** Resize mask with linear interpolation to frame size.  
   **Branch B:** Use mask as-is.  
   **Risk:** Unresized mask would break concat or spatial alignment.

5. **Decision:** Refiner scaling hook  
   **Condition:** `refiner_scale != 1.0` and refiner exists  
   **Branch A:** Register forward hook to scale refiner output logits.  
   **Branch B:** No hook; native deltas.  
   **Risk:** Mis-scaled logits can oversoften or oversharpen matte edges.

6. **Decision:** Postprocess backend  
   **Condition:** `post_process_on_gpu`  
   **Branch A:** Torch/GPU resize, despill, cleanup, composite.  
   **Branch B:** CPU/OpenCV path.  
   **Risk:** Path mismatch yields slightly different interpolation/cleanup behavior.

7. **Decision:** Auto-despeckle  
   **Condition:** `auto_despeckle`  
   **Branch A:** run matte cleanup operators.  
   **Branch B:** raw alpha for processed/composite.  
   **Risk:** false negatives leave dots; false positives remove fine detail.

8. **Decision:** Generate comp preview  
   **Condition:** `generate_comp`  
   **Branch A:** compute checkerboard comp.  
   **Branch B:** `comp=None`.  
   **Risk:** downstream expecting comp may fail if not null-checked.

9. **Decision:** Skip existing frame  
   **Condition:** `skip_existing` and existing comp output  
   **Branch A:** skip inference/write.  
   **Branch B:** process frame.  
   **Risk:** stale outputs if settings changed but outputs reused.

10. **Decision:** Missing alpha hint (user didn’t provide)  
    **Condition:** `alpha_asset is None` during scan/inference eligibility  
    **Branch A:** run generators (GVM/VideoMaMa) first or skip clip.  
    **Branch B:** clip processed only when alpha exists.  
    **Risk:** clip silently excluded from inference batch if user unaware.

11. **Decision:** Frame-count mismatch input vs alpha  
    **Condition:** counts differ  
    **Branch A (clip_manager path):** hard error in `validate_pair` (clip invalid).  
    **Branch B (service path default):** truncate to min count with warning.  
    **Risk:** silent truncation can desync editorial expectations.

12. **Decision:** First-frame temporal smoothing  
    **Condition:** N/A (feature absent)  
    **Branch A/B:** no temporal state branch exists.  
    **Risk:** temporal flicker remains unmitigated.

13. **Decision:** Output writer availability  
    **Condition:** `cv2.imwrite` success, ffmpeg found  
    **Branch A:** write frame / stitch mp4.  
    **Branch B:** warning/error; continue or skip stitch.  
    **Risk:** partial outputs with no hard fail in some paths.

---

### PASS 4 — THE DATA LINEAGE DIAGRAM

```text
[Input Asset]
  ├─ video: OpenCV VideoCapture.read() -> BGR uint8 [H,W,3]
  └─ image: cv2.imread() -> BGR uint8/float [H,W,C]
                 │
                 ▼
[Input Decode + Color Convert]
  BGR/BGRA -> RGB ; uint8 -> float32[0,1] (std images)
  EXR optional linear->sRGB depending on input_is_linear
                 │  RGB float32 [H,W,3]
                 ▼
[Mask Asset]
  ├─ video alpha: frame[:,:,2] / 255
  └─ image alpha: read anydepth + channel/dtype normalize
                 │  mask float32 [H,W]
                 ▼
[Mask/Image Size Align]
  if mask size != image size: resize mask (linear)
                 │
                 ├───────────────┐
                 ▼               ▼
[Tensorization]               [Tensorization]
image -> [B,3,H,W]            mask -> [B,1,H,W]
dtype->model_precision        dtype->model_precision
to(device)                    to(device)
                 └──────┬───────┘
                        ▼
[Preprocess]
resize both to [S,S] (S=2048 default)
if input_is_linear: image linear->sRGB
ImageNet normalize(image)
concat(image,mask) -> [B,4,S,S]
                        │
                        ▼
[GreenFormer]
Hiera encoder(features at 4 scales)
  -> alpha decoder logits [B,1,S/4,S/4]
  -> fg decoder logits    [B,3,S/4,S/4]
upsample logits -> [B,1,S,S], [B,3,S,S]
sigmoid coarse probs
CNN refiner on [RGB + coarse α/FG] -> delta logits [B,4,S,S]
add deltas in logit space
sigmoid final
                        │
                        ├─ alpha [B,1,S,S]
                        └─ fg    [B,3,S,S]
                        ▼
[Resize Back to Original]
alpha, fg -> [B,1,H,W], [B,3,H,W]
                        │
                        ├─ raw alpha export path --------------------> Matte EXR (linear, single-channel)
                        │
                        ▼
[Optional Matte Cleanup]
connected components + area filter + dilation + blur
processed_alpha [B,1,H,W]
                        │
                        ▼
[Despill FG]
on fg sRGB (global per-pixel rule)
fg_despilled_sRGB [B,3,H,W]
                        │
                        ▼
[sRGB -> Linear]
fg_linear [B,3,H,W]
                        │
                        ├─[Comp Preview branch]
                        │    checkerboard_sRGB -> checkerboard_linear
                        │    composite: α*F + (1-α)*BG
                        │    linear->sRGB
                        │    -> comp float [H,W,3] -> quantize uint8 -> Comp PNG
                        │
                        ▼
[Premultiply + Pack]
fg_premul_linear = fg_linear * processed_alpha
processed_rgba = concat(fg_premul_linear, processed_alpha) [H,W,4]
                        │
                        ├─ FG raw branch: fg raw sRGB [H,W,3] -> FG EXR
                        ├─ Matte raw branch: alpha raw [H,W,1] -> Matte EXR
                        └─ Processed branch: processed RGBA linear premul -> Processed EXR
```

---

## 1. SYSTEM OVERVIEW & DESIGN PHILOSOPHY

CorridorKey solves high-fidelity greenscreen extraction where classical keyers fail: semi-transparency, fine hair, motion blur, and color decontamination in mixed pixels. This is non-trivial because each observed pixel is a composite of foreground and background radiance, and one binary threshold cannot recover both alpha and true foreground color.

Why neural matting: the model jointly predicts alpha and foreground, using learned priors to infer plausible foreground color in ambiguous mixed regions. Classical chroma keying is brittle under uneven screens, spill, and soft edges.

Architectural contract between stages:
- Stage A (I/O): always provide RGB float image + linear mask in `[0,1]`.
- Stage B (model): consume normalized 4-channel tensor at fixed square resolution.
- Stage C (post): return raw alpha/fg plus optional processed premultiplied output and checker comp.
- Stage D (writer): emit per-pass artifacts in expected color spaces (FG sRGB, Matte linear, Processed linear premul).

Design constraints visible in code:
- Latency/throughput pressure on video sequences (frame loop).
- Memory pressure (VRAM comments, GPU job serialization lock).
- Batch usually 1 in production loop; engine can accept batched arrays.
- GPU/CPU split selectable for postprocessing.
- Strict color-management concerns (piecewise sRGB conversions, EXR linear contract).

---

## 2. INPUT SPECIFICATION & DATA CONTRACTS

Frame ingestion assumptions:
- Input image expected RGB with shape `[H,W,3]` after decode.
- Video decode path starts BGR uint8.
- EXR accepted as float, can include alpha channel (dropped for input plate).
- Arrays assumed contiguous row-major [INFERRED]; no explicit stride checks.

Green-screen assumptions:
- No explicit chroma-range validator in inference code.
- System relies on hint mask + learned model priors rather than hard green thresholding.
- Uniform lighting not required by code, but quality likely degrades when screen is non-uniform [INFERRED].

Alpha hint contract:
- Dense scalar mask (`[H,W]` or `[H,W,1]`).
- Interpreted as soft confidence/prior, not strict trimap classes.
- Encoded as 4th channel to network input.
- Mismatch in frame count or dimensions is handled via validation/resize/truncation depending call path.

Tensor contracts (core):
- Model input: `[B,4,S,S]`.
- Model outputs: alpha `[B,1,S,S]`, fg `[B,3,S,S]`.
- Post outputs (per-frame dictionaries):  
  `alpha [H,W,1] float32`, `fg [H,W,3] float32`, `comp [H,W,3] float32|None`, `processed [H,W,4] float32`.

Failure modes:
- Loud: missing checkpoint, invalid backend, frame count mismatch (clip-manager path), write failures.
- Soft/silent: wrong gamma flag, skip-existing stale frames, service-mode truncation to min frame count, subtle output drift between CPU/GPU post paths.

---

## 3. THE GREENFORMER NEURAL NETWORK — ARCHITECTURE DEEP DIVE

Model family:
- Transformer-based hybrid matting model.
- Backbone is TIMM Hiera feature extractor (`features_only=True`), plus custom decoders and CNN refiner.

Input tensor construction:
- RGB + alpha-hint concatenated at channel axis.
- First patch embed layer modified 3→4 channels; new channel initialized to zeros and populated by checkpoint weights.

Attention mechanism:
- Implemented inside Hiera (not custom in CorridorKey code).  
- No explicit mask-conditioned attention bias or cross-attention branch coded in this repo.
- Hint influence is implicit via the extra channel entering shared backbone.

Skip/multiscale handling:
- Four encoder feature maps are linearly projected to common embedding dim (256), upsampled to highest decoder scale, concatenated and fused (1x1 conv + BN + ReLU).
- Separate alpha and fg decoders share architecture but not output conv.

Output heads:
- Parallel heads:
  - Alpha head (1 channel logits)
  - FG head (3 channel logits)
- Both upsampled to full resolution before refinement.
- Refiner predicts additive logits (4 channels) conditioned on original RGB + coarse probabilities.
- Final outputs are sigmoid probabilities.

Loss functions:
- Not present in this inference-only repository.  
- Based on outputs and comments, training likely used alpha loss + compositional/color terms + edge/detail terms [INFERRED].

Inference vs training behavior:
- Inference path omits training-only structures; returns only final alpha/fg.
- Uses optional mixed precision and optional `torch.compile`.
- Refiner scaling hook allows runtime strength tuning.

---

## 4. THE ALPHA MATTE EXTRACTION STAGE

Pre-inference preprocessing:
- Resize to fixed square.
- Convert linear→sRGB if flagged.
- Normalize with ImageNet statistics.
- Concatenate hint as 4th channel.

Hint encoding:
- Single-channel dense hint concatenation; no attention bias matrix, no separate hint encoder branch.

Postprocessing:
- Raw alpha is resized back and exported directly.
- Optional cleanup for processed outputs:
  - Connected-component area filtering
  - Dilation
  - Gaussian blur
  - Multiplication with original alpha
- No temporal smoothing across frames.
- No guided filter/detail transfer module.

Unknown-region handling:
- Not explicit trimap classes; unknown/confidence is represented continuously by mask values and learned model behavior.

---

## 5. FOREGROUND RECONSTRUCTION

Foreground generation strategy:
- Foreground is directly model-predicted (`fg` head), not solved via closed-form unmixing from known green background at inference time.
- Therefore classic degenerate division at `α≈0` is avoided.

Matting equation usage:
- Used only in compositing/preview (`αF+(1−α)B`), not for reconstructing `F` from `C`.

Spill suppression:
- Separate algorithmic postprocess (not purely learned).
- Per-channel redistribution based on excess green above `(R+B)/2`.
- Strength parameter allows bypass (0) to full effect (1).

Alpha representation:
- Raw alpha and fg are straight outputs.
- Premultiplication happens only when creating `processed` RGBA (linear EXR-friendly output).

---

## 6. COMPOSITING STAGE

C++ implementation note: this repo’s orchestration is Python; compositing math executes in NumPy/Torch kernels backed by C/CUDA internals.

Compositing equation:
- Straight-over operation: `Out = αF + (1−α)B`.

Background support:
- Internal pipeline supports checkerboard preview background only.
- Arbitrary production backgrounds are expected to be composited downstream in DCC tools using exported passes.

Blend mode:
- Standard over operation only.

Color-space management:
- FG prediction interpreted as sRGB.
- For physically-correct operations:
  - Convert FG and checkerboard to linear.
  - Composite in linear.
  - Convert preview back to sRGB.
- Processed EXR remains linear premultiplied.

Edge anti-aliasing/feathering:
- Matte cleanup blur acts as soft feather.
- No dedicated subpixel edge AA stage.

---

## 7. C++ IMPLEMENTATION ARCHITECTURE

There is no direct C++ application layer in this repository; implementation is Python + PyTorch/OpenCV/FFmpeg. The closest “systems architecture” equivalent:

- **Threading model**
  - CLI path: mostly single-threaded frame loop.
  - Service path: explicit GPU mutex (`threading.Lock`) serializes all model operations.
  - Job queue supports one GPU job at a time, with cancellation and progress callbacks.

- **Memory management**
  - CPU decode buffers in NumPy.
  - NumPy→Torch tensor copies each frame.
  - GPU↔CPU transfers occur when returning outputs to NumPy (`.cpu().numpy()`).
  - No explicit pinned memory pools [INFERRED], no explicit reusable tensor pool.

- **Frame buffering**
  - Sequential processing; no deep pipeline overlap decode/infer/write.
  - Optional skip/resume behavior by stem checks.
  - No ring buffer/double buffering.

- **Tensor ownership**
  - Engine owns model/tensors.
  - Caller owns input NumPy arrays and output dictionaries.
  - Intermediate tensors freed by scope + explicit `del` in some hot spots.

- **Resource lifecycle**
  - VideoCapture opened per clip and released in finally blocks.
  - Model loaded once per engine creation.
  - `torch.compile` optional with fallback.

- **Error propagation**
  - Mixed approach:
    - Typed exceptions in backend service (`CorridorKeyError` subclasses).
    - Warning/continue behavior in clip_manager frame loop.
    - Non-fatal ffmpeg stitch failures logged and ignored.

---

## 8. PERFORMANCE CHARACTERISTICS & BOTTLENECK MAP

Per-stage latency shape (qualitative):
- Dominant: model inference at 2048².
- Secondary: resize + postprocess + CPU<->GPU transfer.
- Significant on HDD/network: EXR writes.

Bandwidth pressure points:
- Full-frame 2048² tensors (4-channel input + outputs).
- Repeated up/downsampling.
- GPU->CPU copies for all outputs in current implementation.
- EXR write throughput and compression.

GPU utilization under-saturation points:
- Single-frame sequential processing (batch=1 typical).
- CPU-side decode/write not overlapped with GPU inference.
- Optional CPU postprocess path introduces stalls.
- Torch compile warmup cost at first run.

Known regressions / stress conditions:
- 16GB VRAM systems can OOM at 2048² (noted in docs/comments).
- Wrong gamma flag causes visual artifacts (“dark fringes”, crushed tones).
- Poor hint masks propagate errors directly.
- Fast thin detail may flicker due to no temporal model.

Profiling hooks:
- Service logs per-frame process time (`time.monotonic()` debug).
- No integrated profiler timeline (no NVTX or structured stage timings).

---

## 9. REUSABLE TECHNIQUES & PATTERNS (EXTRACTION GUIDE)

### Technique Card 1 — Hint-as-4th-Channel Conditioning
- **Problem solved:** Inject user guidance with minimal architecture complexity.
- **Mechanics:** Concatenate mask to RGB; patch first conv to 4 channels.
- **Where:** `GreenFormer._patch_input_layer`, engine preprocessing.
- **Reuse:** Any segmentation/matting model can adopt this with weight patching.
- **Variants:** Late-fusion hint branch, cross-attention hint tokens, trimap embedding.

### Technique Card 2 — Logit-Space Refinement Head
- **Problem solved:** Correct coarse model artifacts without saturating outputs.
- **Mechanics:** Predict delta logits from RGB + coarse probs; add before sigmoid.
- **Where:** `CNNRefinerModule`, `GreenFormer.forward`.
- **Reuse:** Useful in depth/mask refinement where probability saturation is an issue.
- **Variants:** UNet refiner, transformer refiner, iterative residual refinement.

### Technique Card 3 — Color-Managed EXR Packaging
- **Problem solved:** Avoid gamma/premul mistakes in VFX interchange.
- **Mechanics:** sRGB FG → linear → premultiply with linear alpha → EXR half-float.
- **Where:** `inference_engine` postprocess + `color_utils`.
- **Reuse:** Any VFX pipeline needing faithful linear deliverables.
- **Variants:** ACEScg path, deep EXR, float32 full precision.

### Technique Card 4 — Lightweight Despill with Energy Redistribution
- **Problem solved:** Remove green contamination while preserving apparent brightness.
- **Mechanics:** Clamp excess green and redistribute to R/B.
- **Where:** `color_utils.despill_*`.
- **Reuse:** Real-time keying previews, post matte cleanup.
- **Variants:** YUV-domain despill, hue-rotation despill, ML-based decontamination.

### Technique Card 5 — Connected-Component Matte Cleaning
- **Problem solved:** Remove small false-positive islands/tracking dots.
- **Mechanics:** Threshold, CC filter by area, dilate, blur, multiply with alpha.
- **Where:** `clean_matte_opencv`, `clean_matte_torch`.
- **Reuse:** Any binary/soft mask denoising stage.
- **Variants:** Morphological opening/closing, CRF cleanup, learned refinement.

### Technique Card 6 — Backend Contract Normalization (Torch/MLX)
- **Problem solved:** Keep one downstream API despite heterogeneous runtimes.
- **Mechanics:** Adapter wraps MLX uint8 outputs into Torch float contract and applies missing post steps.
- **Where:** `CorridorKeyModule/backend.py`.
- **Reuse:** Multi-runtime inference products with common post stack.
- **Variants:** ONNX/TensorRT adapters, backend capability negotiation.

### Technique Card 7 — GPU Job Serialization for VRAM Safety
- **Problem solved:** Prevent OOM/race from concurrent heavy model runs.
- **Mechanics:** Global GPU mutex + single-consumer job queue + cancel flags.
- **Where:** `backend/job_queue.py`, `backend/service.py`.
- **Reuse:** Desktop tools running multiple GPU pipelines.
- **Variants:** Weighted fair scheduler, per-model VRAM budgeting.

### Technique Card 8 — Streaming Video Inference with Stem-Based Resume
- **Problem solved:** Recover from interrupted long renders.
- **Mechanics:** Determine frame stem, skip if output exists, continue.
- **Where:** `clip_manager.run_inference`, service equivalents.
- **Reuse:** Any sequence render pipeline.
- **Variants:** manifest-hash-based resume, partial-stage resume.

---

## 10. KNOWN LIMITATIONS & FAILURE CASES

- Uneven greenscreen lighting and color contamination reduce reliability (hint helps but does not fully correct).
- Motion blur/fine detail can still fail in highly ambiguous regions.
- Strong dependence on hint quality; over-expanded hints leak foreground assumptions.
- No temporal model means frame-to-frame alpha instability is possible.
- Background support in internal compositor is only checkerboard; production comp externalized.
- Aspect-ratio distortion due to square resize may hurt composition for extreme formats.
- Global despill can over-correct legitimate green wardrobe/props.
- Classical chroma key can outperform in clean studio setups requiring ultra-low latency and deterministic controls.

---

## 11. OPTIMIZATION ROADMAP (FUTURE WORK)

1. **Model quantization (INT8/FP16 tightening)**
   - **Why:** reduce inference latency/VRAM.
   - **Complexity:** Medium-High (calibration + quality regression management).

2. **Optical-flow-guided temporal consistency**
   - **Why:** reduce flicker and edge shimmer.
   - **Complexity:** High (flow estimation + occlusion-aware fusion).

3. **Sparse hint upsampling via learned propagation**
   - **Why:** support sparse user strokes, reduce mask prep burden.
   - **Complexity:** High (new model branch/training).

4. **Replace heuristic matte cleanup with learned refinement network**
   - **Why:** better detail retention than CC+dilate+blur heuristics.
   - **Complexity:** Medium-High.

5. **Batched multi-frame inference with temporal context**
   - **Why:** higher GPU occupancy + temporal coherence.
   - **Complexity:** High (memory planning + model redesign).

6. **CUDA/Torch kernel fusion in pre/post stack**
   - **Why:** reduce memory traffic and launch overhead.
   - **Complexity:** Medium.

7. **Decode-infer-write overlap**
   - **Why:** improve throughput by pipelining CPU I/O and GPU compute.
   - **Complexity:** Medium.

8. **Aspect-ratio-preserving inference (letterbox + unletterbox)**
   - **Why:** avoid geometric distortion artifacts.
   - **Complexity:** Medium.

---

## 12. GLOSSARY OF DOMAIN TERMS

- **Alpha matte:** Per-pixel opacity map in `[0,1]`.
- **Straight alpha:** RGB not multiplied by alpha.
- **Premultiplied alpha:** RGB already multiplied by alpha.
- **Trimap:** 3-class mask (FG/BG/Unknown); not explicitly used here.
- **Hint mask (coarse alpha hint):** User/proxy soft mask guiding model attention implicitly via input channel.
- **Chroma key:** Traditional keying via color thresholds.
- **Spill suppression (despill):** Removing background color contamination from foreground.
- **Matting equation:** `C = αF + (1−α)B`.
- **Foreground reconstruction:** Estimating `F`; here directly predicted by model.
- **Compositing (over operation):** `Out = αF + (1−α)B`.
- **Linear color space:** Radiometrically linear domain for physically correct blending.
- **sRGB transfer function:** Piecewise nonlinear encoding/decoding between display and linear space.
- **ImageNet normalization:** Channel-wise `(x-mean)/std` normalization with canonical values.
- **Patch embedding:** Initial conv/projection turning pixels into token/feature space.
- **Feature pyramid/multiscale fusion:** Combining backbone features from multiple resolutions.
- **Logit:** Pre-sigmoid score in `(-∞,+∞)`.
- **Autocast mixed precision:** Runtime casting ops to lower precision (fp16/bf16) where safe.
- **Connected components:** Labeling contiguous binary regions for morphology filtering.
- **Dilation:** Morphological expansion of mask regions.
- **Gaussian blur:** Smoothing filter for soft edges.
- **PXR24 EXR compression:** OpenEXR compression mode used with half-float storage.
- **BCHW / HWC:** Tensor memory layouts (`Batch,Channel,Height,Width` vs `Height,Width,Channel`).
- **VRAM:** GPU memory.
- **OOM:** Out-of-memory failure.
- **Backend adapter:** Layer that normalizes outputs across runtimes (Torch/MLX).
- **Frame stem:** Filename base used for pairing and output naming.
- **Resume rendering:** Skipping already-rendered frames to continue interrupted runs.

---

### Final note on scope fidelity
The repository does **not** contain a native C++ pipeline implementation; it contains Python orchestration over C/CUDA-backed libraries (PyTorch/OpenCV/FFmpeg). All C++-layer behavior above is either directly observed via these APIs or explicitly marked [INFERRED] where implementation internals are hidden in upstream libraries.
