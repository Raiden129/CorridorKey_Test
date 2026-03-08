"""VRAM-optimized inference engine.

Drop-in replacement for CorridorKeyEngine.  Uses OptimizedGreenFormer with
tiled CNN refiner and memory hygiene to reduce peak VRAM from ~22.7 GB to
under 8 GB.  Optional hint-based token routing can be enabled for further
savings (requires fine-tuning).
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core import color_utils as cu
from .core.optimized_model import OptimizedGreenFormer


class OptimizedCorridorKeyEngine:
    """Inference engine with VRAM optimizations.

    API-compatible with CorridorKeyEngine.  Same constructor signature
    (plus extra tuning knobs), identical ``process_frame()`` output contract.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        # Token routing (OFF by default -- needs fine-tuning)
        use_token_routing: bool = False,
        edge_threshold_low: float = 0.02,
        edge_threshold_high: float = 0.98,
        min_edge_tokens: int = 64,
        # Tiling knobs
        tile_size: int = 512,
        tile_overlap: int = 128,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Memory optimizations
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            print("[Optimized] cuDNN benchmark disabled (saves 2-5 GB workspace).")

        self.model = self._load_model(
            use_token_routing=use_token_routing,
            edge_threshold_low=edge_threshold_low,
            edge_threshold_high=edge_threshold_high,
            min_edge_tokens=min_edge_tokens,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

    def _load_model(
        self,
        use_token_routing: bool,
        edge_threshold_low: float,
        edge_threshold_high: float,
        min_edge_tokens: int,
        tile_size: int,
        tile_overlap: int,
    ) -> OptimizedGreenFormer:
        print(f"[Optimized] Loading checkpoint from {self.checkpoint_path}...")

        # Verify SDPA availability (guaranteed for PyTorch 2.0+)
        assert hasattr(F, "scaled_dot_product_attention"), (
            "PyTorch 2.0+ is required for F.scaled_dot_product_attention (FlashAttention). "
            f"Current version: {torch.__version__}"
        )

        model = OptimizedGreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=self.img_size,
            use_refiner=self.use_refiner,
            use_token_routing=use_token_routing,
            edge_threshold_low=edge_threshold_low,
            edge_threshold_high=edge_threshold_high,
            min_edge_tokens=min_edge_tokens,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        model = model.to(self.device)
        model.eval()

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Fix compiled-model prefix & handle PosEmbed mismatch
        # (identical logic to CorridorKeyEngine)
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]

            # PosEmbed interpolation
            if "pos_embed" in k and k in model_state:
                if v.shape != model_state[k].shape:
                    print(f"[Optimized] Resizing {k} from {v.shape} to {model_state[k].shape}")
                    N_src = v.shape[1]
                    C = v.shape[2]
                    grid_src = int(math.sqrt(N_src))
                    N_dst = model_state[k].shape[1]
                    grid_dst = int(math.sqrt(N_dst))
                    v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                    v_resized = F.interpolate(
                        v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False
                    )
                    v = v_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        # LTRM keys are expected to be missing (they're new modules)
        ltrm_missing = [k for k in missing if "ltrm" in k]
        other_missing = [k for k in missing if "ltrm" not in k]

        if ltrm_missing:
            print(f"[Optimized] Expected new LTRM keys not in checkpoint ({len(ltrm_missing)} keys) -- using zero-init.")
        if other_missing:
            print(f"[Warning] Missing non-LTRM keys: {other_missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected}")

        # Report VRAM usage after loading
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            print(f"[Optimized] Model loaded. GPU memory: {allocated:.2f} GB")

        return model

    @torch.no_grad()
    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        """Process a single frame.

        API-identical to CorridorKeyEngine.process_frame().
        """
        # 1. Input normalization
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # 2. Resize to model size
        if input_is_linear:
            img_resized_lin = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # 3. ImageNet normalization
        img_norm = (img_resized - self.mean) / self.std

        # 4. Prepare tensor
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1)  # [H, W, 4]
        inp_t = torch.from_numpy(inp_np.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)

        # 5. Inference
        handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:

            def scale_hook(module, input, output):
                return output * refiner_scale

            handle = self.model.refiner.register_forward_hook(scale_hook)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            out = self.model(inp_t)

        if handle:
            handle.remove()

        pred_alpha = out["alpha"]
        pred_fg = out["fg"]

        # 6. Post-process (resize back to original resolution)
        res_alpha = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
        res_fg = pred_fg[0].permute(1, 2, 0).float().cpu().numpy()
        res_alpha = cv2.resize(res_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_fg = cv2.resize(res_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]

        # --- Advanced Compositing ---

        # A. Clean matte
        if auto_despeckle:
            processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = res_alpha

        # B. Despill FG
        fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=despill_strength)

        # C. Premultiply for EXR
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)

        # D. Pack RGBA
        processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

        # 7. Composite on checkerboard
        bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        bg_lin = cu.srgb_to_linear(bg_srgb)

        if fg_is_straight:
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        else:
            comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

        comp_srgb = cu.linear_to_srgb(comp_lin)

        # Report peak VRAM
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            print(f"[Optimized] Peak VRAM: {peak:.2f} GB")

        return {
            "alpha": res_alpha,
            "fg": res_fg,
            "comp": comp_srgb,
            "processed": processed_rgba,
        }
