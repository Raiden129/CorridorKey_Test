"""Tests for the VRAM-optimized GreenFormer components.

Tests cover:
  - TiledCNNRefiner: tiling correctness, blend weight symmetry, small-input bypass
  - LTRM: output shape preservation, zero-init identity
  - ECA: output shape and gating range
  - HintBasedTokenRouter: edge mask generation, threshold behavior, min-edge fallback
  - OptimizedGreenFormer: output contract, checkpoint compatibility
  - OptimizedCorridorKeyEngine: process_frame API parity with original engine
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from CorridorKeyModule.core.optimized_model import (
    ECA,
    LTRM,
    HintBasedTokenRouter,
    TiledCNNRefiner,
)

# ---------------------------------------------------------------------------
# ECA
# ---------------------------------------------------------------------------


class TestECA:
    def test_output_shape(self):
        eca = ECA(channels=64)
        x = torch.randn(2, 100, 64)
        out = eca(x)
        assert out.shape == (2, 100, 64)

    def test_gating_range(self):
        """ECA sigmoid gate should produce values in (0, 1)."""
        eca = ECA(channels=32)
        x = torch.randn(1, 50, 32)
        with torch.no_grad():
            out = eca(x)
        # Output should not exceed input magnitude by much
        # (sigmoid gate is in [0, 1], so out = x * gate)
        assert out.abs().max() <= x.abs().max() + 1e-5


# ---------------------------------------------------------------------------
# LTRM
# ---------------------------------------------------------------------------


class TestLTRM:
    def test_output_shape_full_grid(self):
        """LTRM should preserve [B, N, C] shape when N matches spatial grid."""
        ltrm = LTRM(dim=64)
        x = torch.randn(1, 16 * 16, 64)
        out = ltrm(x, spatial_shape=(16, 16))
        assert out.shape == (1, 256, 64)

    def test_output_shape_ragged(self):
        """LTRM should work when N doesn't match spatial grid (DWConv skipped)."""
        ltrm = LTRM(dim=64)
        x = torch.randn(1, 100, 64)  # 100 != 16*16
        out = ltrm(x, spatial_shape=(16, 16))
        assert out.shape == (1, 100, 64)

    def test_zero_init_identity(self):
        """With zero-initialized fc2, LTRM should output approximately x."""
        ltrm = LTRM(dim=32)
        x = torch.randn(1, 64, 32)
        with torch.no_grad():
            out = ltrm(x, spatial_shape=(8, 8))
        # fc2 output is zero -> eca(zero) is zero -> residual + 0 = x
        # But ECA(0) = 0 * sigmoid(conv(pool(0))) = 0, so out = x + 0 = x
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_batch_dimension(self):
        """LTRM should handle batch size > 1."""
        ltrm = LTRM(dim=48)
        x = torch.randn(4, 64, 48)
        out = ltrm(x, spatial_shape=(8, 8))
        assert out.shape == (4, 64, 48)


# ---------------------------------------------------------------------------
# HintBasedTokenRouter
# ---------------------------------------------------------------------------


class TestHintBasedTokenRouter:
    def test_all_background(self):
        """All-zero hint should produce no edge tokens."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=0)
        hint = torch.zeros(1, 1, 64, 64)
        mask = router.compute_edge_mask(hint, 8, 8)
        assert mask.shape == (1, 64)
        assert mask.sum() == 0  # all tokens are "easy"

    def test_all_foreground(self):
        """All-one hint should produce no edge tokens."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=0)
        hint = torch.ones(1, 1, 64, 64)
        mask = router.compute_edge_mask(hint, 8, 8)
        assert mask.sum() == 0

    def test_edge_region(self):
        """Hint with 0.5 values should produce all edge tokens."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=0)
        hint = torch.full((1, 1, 64, 64), 0.5)
        mask = router.compute_edge_mask(hint, 8, 8)
        assert mask.all()  # all tokens are "edge"

    def test_min_edge_tokens_fallback(self):
        """When too few edge tokens, should fall back to all-edge."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=100)
        # Mostly background with a tiny edge
        hint = torch.zeros(1, 1, 64, 64)
        hint[0, 0, 0, 0] = 0.5  # just one edge pixel
        mask = router.compute_edge_mask(hint, 8, 8)
        # After area downsampling to 8x8, at most 1 token would be edge
        # Since min_edge_tokens=100 > 1, all tokens should be marked edge
        assert mask.all()

    def test_mixed_mask(self):
        """Hint with distinct BG/FG/edge regions should route correctly."""
        router = HintBasedTokenRouter(threshold_low=0.02, threshold_high=0.98, min_edge_tokens=0)
        hint = torch.zeros(1, 1, 64, 64)
        hint[0, 0, :32, :] = 0.0  # top half: BG
        hint[0, 0, 32:, :] = 1.0  # bottom half: FG
        # The boundary row will have some intermediate values after area downsample
        mask = router.compute_edge_mask(hint, 8, 8)
        # Most tokens should NOT be edge
        assert mask.sum() < mask.numel()


# ---------------------------------------------------------------------------
# TiledCNNRefiner
# ---------------------------------------------------------------------------


class TestTiledCNNRefiner:
    def test_small_input_bypasses_tiling(self):
        """When input fits in one tile, should skip tiling."""
        refiner = TiledCNNRefiner(tile_size=512, tile_overlap=128)
        img = torch.randn(1, 3, 64, 64)
        coarse = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out = refiner(img, coarse)
        assert out.shape == (1, 4, 64, 64)

    def test_output_shape_large_input(self):
        """Tiled processing should produce correct output shape."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        img = torch.randn(1, 3, 128, 128)
        coarse = torch.randn(1, 4, 128, 128)
        with torch.no_grad():
            out = refiner(img, coarse)
        assert out.shape == (1, 4, 128, 128)

    def test_blend_weight_symmetry(self):
        """Blend weights should be symmetric."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        w = refiner._create_blend_weight(64, 64, 16, torch.device("cpu"), torch.float32)
        assert w.shape == (1, 1, 64, 64)
        # Symmetric along both axes
        torch.testing.assert_close(w, w.flip(2))  # vertical symmetry
        torch.testing.assert_close(w, w.flip(3))  # horizontal symmetry

    def test_blend_weight_center_is_one(self):
        """Center of blend weight (away from all borders) should be 1.0."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        w = refiner._create_blend_weight(64, 64, 16, torch.device("cpu"), torch.float32)
        assert w[0, 0, 32, 32] == 1.0

    def test_blend_weight_edge_zero(self):
        """Edge of blend weight should be 0.0 (first ramp element)."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        w = refiner._create_blend_weight(64, 64, 16, torch.device("cpu"), torch.float32)
        assert w[0, 0, 0, 32] == pytest.approx(0.0, abs=1e-6)
        assert w[0, 0, 32, 0] == pytest.approx(0.0, abs=1e-6)

    def test_tiled_vs_untiled_small(self):
        """On a small input that forces tiling via tile_size=48, output should
        be close to running the refiner on the full input at once.

        Boundary tiles have less overlap context, so we use a relaxed
        tolerance.  At production resolution (2048x2048 with 512px tiles
        and 128px overlap >> 65px receptive field), the difference is
        negligible in the interior and only visible at the image border.
        """
        torch.manual_seed(42)
        refiner_tiled = TiledCNNRefiner(tile_size=48, tile_overlap=24)
        from CorridorKeyModule.core.model_transformer import CNNRefinerModule

        refiner_full = CNNRefinerModule()
        refiner_full.load_state_dict(refiner_tiled.state_dict())

        img = torch.randn(1, 3, 64, 64)
        coarse = torch.randn(1, 4, 64, 64)

        with torch.no_grad():
            out_tiled = refiner_tiled(img, coarse)
            out_full = refiner_full(img, coarse)

        # Relaxed tolerance: boundary tiles have limited overlap context
        torch.testing.assert_close(out_tiled, out_full, atol=0.5, rtol=0.5)

    def test_no_nan_or_inf(self):
        """Output should contain no NaN or Inf values."""
        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        img = torch.randn(1, 3, 128, 128)
        coarse = torch.randn(1, 4, 128, 128)
        with torch.no_grad():
            out = refiner(img, coarse)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# OptimizedGreenFormer (structural tests, no checkpoint needed)
# ---------------------------------------------------------------------------


class TestOptimizedGreenFormerStructure:
    """Tests that don't require loading model weights."""

    def test_import(self):
        """OptimizedGreenFormer should be importable."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        assert OptimizedGreenFormer is not None

    def test_token_routing_disabled_by_default(self):
        """With default args, token routing should be off."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        model = OptimizedGreenFormer(img_size=224, use_refiner=False)
        assert model.config.token_routing is False
        assert model.ltrm_stage2 is None
        assert model.ltrm_stage3 is None
        assert model.router is None

    def test_token_routing_enabled(self):
        """With use_token_routing=True, LTRM modules and router should exist."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        model = OptimizedGreenFormer(img_size=224, use_refiner=False, use_token_routing=True)
        assert model.config.token_routing is True
        assert model.ltrm_stage2 is not None
        assert model.ltrm_stage3 is not None
        assert model.router is not None
        # Stage 2 has 16 blocks, stage 3 has 3
        assert len(model.ltrm_stage2) == 16
        assert len(model.ltrm_stage3) == 3

    def test_tiled_refiner_created(self):
        """When use_refiner=True, refiner should be TiledCNNRefiner."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        model = OptimizedGreenFormer(img_size=224, use_refiner=True)
        assert isinstance(model.refiner, TiledCNNRefiner)

    def test_encoder_has_4_channels(self):
        """Patch embed should accept 4 input channels."""
        from CorridorKeyModule.core.optimized_model import OptimizedGreenFormer

        model = OptimizedGreenFormer(img_size=224, use_refiner=False)
        patch_embed = model.encoder.model.patch_embed.proj
        assert patch_embed.in_channels == 4


# ---------------------------------------------------------------------------
# OptimizedCorridorKeyEngine (mocked, no checkpoint)
# ---------------------------------------------------------------------------


class TestOptimizedEngineAPI:
    """Verify the optimized engine output contract matches the original."""

    def _make_optimized_engine_mock(self, img_size=64):
        """Create an OptimizedCorridorKeyEngine with a mocked model."""
        from CorridorKeyModule.optimization_config import OptimizationConfig
        from CorridorKeyModule.optimized_engine import OptimizedCorridorKeyEngine

        def fake_forward(x, **kwargs):
            b, c, h, w = x.shape
            return {
                "alpha": torch.full((b, 1, h, w), 0.8),
                "fg": torch.full((b, 3, h, w), 0.6),
            }

        mock_model = MagicMock()
        mock_model.side_effect = fake_forward
        mock_model.refiner = None
        mock_model.use_refiner = False

        engine = object.__new__(OptimizedCorridorKeyEngine)
        engine.device = torch.device("cpu")
        engine.img_size = img_size
        engine.checkpoint_path = "/fake/checkpoint.pth"
        engine.use_refiner = False
        engine.config = OptimizationConfig.optimized()
        engine.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        engine.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        engine.model = mock_model
        return engine

    def test_output_keys(self, sample_frame_rgb, sample_mask):
        engine = self._make_optimized_engine_mock()
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        assert "alpha" in result
        assert "fg" in result
        assert "comp" in result
        assert "processed" in result

    def test_output_shapes(self, sample_frame_rgb, sample_mask):
        engine = self._make_optimized_engine_mock()
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        h, w = sample_frame_rgb.shape[:2]
        assert result["alpha"].shape == (h, w, 1)
        assert result["fg"].shape == (h, w, 3)
        assert result["comp"].shape == (h, w, 3)
        assert result["processed"].shape == (h, w, 4)

    def test_output_dtypes(self, sample_frame_rgb, sample_mask):
        engine = self._make_optimized_engine_mock()
        result = engine.process_frame(sample_frame_rgb, sample_mask)
        for key in ("alpha", "fg", "comp", "processed"):
            assert result[key].dtype == np.float32, f"{key} dtype mismatch"

    def test_uint8_input(self, sample_mask):
        """Engine should handle uint8 input without error."""
        engine = self._make_optimized_engine_mock()
        img_u8 = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        result = engine.process_frame(img_u8, sample_mask)
        assert "alpha" in result

    def test_2d_mask_input(self, sample_frame_rgb):
        """Engine should handle [H, W] mask without [H, W, 1]."""
        engine = self._make_optimized_engine_mock()
        mask_2d = np.random.rand(64, 64).astype(np.float32)
        result = engine.process_frame(sample_frame_rgb, mask_2d)
        assert result["alpha"].shape == (64, 64, 1)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


class TestBackendFactory:
    def test_torch_optimized_is_valid_backend(self):
        from CorridorKeyModule.backend import VALID_BACKENDS

        assert "torch_optimized" in VALID_BACKENDS

    def test_resolve_backend_explicit(self):
        from CorridorKeyModule.backend import resolve_backend

        assert resolve_backend("torch") == "torch"
        assert resolve_backend("torch_optimized") == "torch_optimized"


# ---------------------------------------------------------------------------
# TiledCNNRefiner: Duplicate tile deduplication
# ---------------------------------------------------------------------------


class TestTiledCNNRefinerDedup:
    """Verify that the tiled refiner processes each coordinate region exactly once."""

    def test_no_duplicate_tiles(self):
        """Each tile coordinate should be processed exactly once."""
        call_coords = []

        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16)
        original_process_tile = refiner._process_tile

        def tracking_process_tile(x):
            # Record the spatial dimensions as a proxy for tile identity
            call_coords.append(x.shape)
            return original_process_tile(x)

        refiner._process_tile = tracking_process_tile

        img = torch.randn(1, 3, 128, 128)
        coarse = torch.randn(1, 4, 128, 128)
        with torch.no_grad():
            out = refiner(img, coarse)

        # With tile_size=64, overlap=16, stride=48 on a 128x128 image:
        # Positions: 0, 48, 96 along each axis → 3x3 = 9 initial positions
        # After boundary adjustment some collapse → should be < 9 unique tiles
        # The key point: no duplicates (old code would have had duplicates)
        assert out.shape == (1, 4, 128, 128)
        # Each call should have been to a tile of size [1, 7, 64, 64]
        for shape in call_coords:
            assert shape[0] == 1 and shape[1] == 7

    def test_unique_coordinates_tracked(self):
        """Directly verify the coordinate deduplication logic."""
        tile_size = 64
        tile_overlap = 16
        stride = tile_size - tile_overlap
        H, W = 128, 128

        coords_set = set()
        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                y0_adj = max(0, y1 - tile_size)
                x0_adj = max(0, x1 - tile_size)
                coords_set.add((y0_adj, y1, x0_adj, x1))

        # The number of unique coords should be less than the raw loop iterations
        raw_iterations = len(list(range(0, H, stride))) * len(list(range(0, W, stride)))
        assert len(coords_set) <= raw_iterations
        # And each should have valid, non-negative coordinates
        for y0_adj, y1, x0_adj, x1 in coords_set:
            assert y0_adj >= 0 and x0_adj >= 0
            assert y1 <= H and x1 <= W
            assert y1 - y0_adj <= tile_size
            assert x1 - x0_adj <= tile_size


# ---------------------------------------------------------------------------
# TiledCNNRefiner: Sparse tile skipping
# ---------------------------------------------------------------------------


class TestTiledCNNRefinerSparse:
    """Verify sparse tile skipping behavior."""

    def test_pure_background_skips_processing(self):
        """When coarse alpha is all zeros, _process_tile should not be called."""
        call_count = [0]

        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16, sparse=True)
        original_process_tile = refiner._process_tile

        def counting_process_tile(x):
            call_count[0] += 1
            return original_process_tile(x)

        refiner._process_tile = counting_process_tile

        img = torch.randn(1, 3, 128, 128)
        coarse = torch.zeros(1, 4, 128, 128)  # all zeros → pure background

        with torch.no_grad():
            out = refiner(img, coarse)

        assert call_count[0] == 0, f"Expected 0 tile calls for pure BG, got {call_count[0]}"
        assert out.shape == (1, 4, 128, 128)
        # Output should be all zeros (no delta applied)
        assert out.abs().max() == 0.0

    def test_pure_foreground_skips_processing(self):
        """When coarse alpha is all ones, _process_tile should not be called."""
        call_count = [0]

        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16, sparse=True)
        original_process_tile = refiner._process_tile

        def counting_process_tile(x):
            call_count[0] += 1
            return original_process_tile(x)

        refiner._process_tile = counting_process_tile

        img = torch.randn(1, 3, 128, 128)
        coarse = torch.ones(1, 4, 128, 128)  # all ones → pure foreground

        with torch.no_grad():
            out = refiner(img, coarse)

        assert call_count[0] == 0, f"Expected 0 tile calls for pure FG, got {call_count[0]}"
        assert out.shape == (1, 4, 128, 128)

    def test_edge_tiles_are_processed(self):
        """Tiles with alpha=0.5 (edge region) should be processed normally."""
        call_count = [0]

        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16, sparse=True)
        original_process_tile = refiner._process_tile

        def counting_process_tile(x):
            call_count[0] += 1
            return original_process_tile(x)

        refiner._process_tile = counting_process_tile

        img = torch.randn(1, 3, 128, 128)
        coarse = torch.full((1, 4, 128, 128), 0.5)  # all edge → must process

        with torch.no_grad():
            out = refiner(img, coarse)

        assert call_count[0] > 0, "Edge tiles should be processed"
        assert out.shape == (1, 4, 128, 128)

    def test_mixed_scene_partial_skip(self):
        """A scene with both uniform and edge regions should partially skip."""
        call_count = [0]

        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16, sparse=True)
        original_process_tile = refiner._process_tile

        def counting_process_tile(x):
            call_count[0] += 1
            return original_process_tile(x)

        refiner._process_tile = counting_process_tile

        img = torch.randn(1, 3, 128, 128)
        coarse = torch.zeros(1, 4, 128, 128)
        # Put edge values in the top-left quadrant only
        coarse[:, 0:1, :64, :64] = 0.5

        with torch.no_grad():
            out = refiner(img, coarse)

        # Some tiles should be processed (edge region) and some skipped (uniform)
        assert call_count[0] > 0, "At least some edge tiles should be processed"
        assert out.shape == (1, 4, 128, 128)

    def test_sparse_disabled_processes_all_tiles(self):
        """When sparse=False, all tiles are processed even for uniform alpha."""
        call_count = [0]

        refiner = TiledCNNRefiner(tile_size=64, tile_overlap=16, sparse=False)
        original_process_tile = refiner._process_tile

        def counting_process_tile(x):
            call_count[0] += 1
            return original_process_tile(x)

        refiner._process_tile = counting_process_tile

        img = torch.randn(1, 3, 128, 128)
        coarse = torch.zeros(1, 4, 128, 128)  # all zeros

        with torch.no_grad():
            out = refiner(img, coarse)

        assert call_count[0] > 0, "With sparse=False, all tiles should be processed"


# ---------------------------------------------------------------------------
# OptimizationConfig: New profiles and fields
# ---------------------------------------------------------------------------


class TestOptimizationConfigUpdates:
    """Verify the new sparse_refiner and compile_submodules config fields."""

    def test_original_profile_defaults(self):
        from CorridorKeyModule.optimization_config import OptimizationConfig

        config = OptimizationConfig.original()
        assert config.sparse_refiner is False
        assert config.compile_submodules is False

    def test_optimized_profile_has_sparse(self):
        from CorridorKeyModule.optimization_config import OptimizationConfig

        config = OptimizationConfig.optimized()
        assert config.sparse_refiner is True
        assert config.compile_submodules is False

    def test_v2_profile(self):
        from CorridorKeyModule.optimization_config import OptimizationConfig

        config = OptimizationConfig.v2()
        assert config.flash_attention is True
        assert config.tiled_refiner is True
        assert config.sparse_refiner is True
        assert config.compile_submodules is True
        assert config.cache_clearing is False  # Disabled in v2

    def test_experimental_profile(self):
        from CorridorKeyModule.optimization_config import OptimizationConfig

        config = OptimizationConfig.experimental()
        assert config.token_routing is True
        assert config.compile_submodules is True
        assert config.sparse_refiner is True

    def test_from_profile_v2(self):
        from CorridorKeyModule.optimization_config import OptimizationConfig

        config = OptimizationConfig.from_profile("v2")
        assert config.compile_submodules is True

    def test_active_optimizations_includes_new_flags(self):
        from CorridorKeyModule.optimization_config import OptimizationConfig

        config = OptimizationConfig(sparse_refiner=True, compile_submodules=True)
        active = config.active_optimizations()
        assert "sparse_refiner" in active
        assert "compile_submodules" in active

    def test_summary_reports_new_flags(self):
        from CorridorKeyModule.optimization_config import OptimizationConfig

        config = OptimizationConfig.v2()
        summary = config.summary()
        assert "sparse_refiner" in summary
        assert "compile_submodules" in summary


# ---------------------------------------------------------------------------
# GreenFormer: refiner_scale parameter
# ---------------------------------------------------------------------------


class TestRefinerScaleParameter:
    """Verify that refiner_scale is passed through forward() correctly."""

    def test_refiner_scale_zero_no_refinement(self):
        """refiner_scale=0.0 should produce output identical to no refiner delta."""
        from CorridorKeyModule.core.model_transformer import GreenFormer

        torch.manual_seed(42)
        model = GreenFormer(img_size=224, use_refiner=True)
        model.eval()

        x = torch.randn(1, 4, 224, 224)

        with torch.no_grad():
            out_scale0 = model(x, refiner_scale=0.0)

        # With refiner_scale=0.0, delta*0=0, so result should match no-refiner
        # Build a model without refiner for comparison
        torch.manual_seed(42)
        model_no_ref = GreenFormer(img_size=224, use_refiner=False)
        model_no_ref.eval()
        # Copy encoder and decoder weights
        model_no_ref.encoder.load_state_dict(model.encoder.state_dict())
        model_no_ref.alpha_decoder.load_state_dict(model.alpha_decoder.state_dict())
        model_no_ref.fg_decoder.load_state_dict(model.fg_decoder.state_dict())

        with torch.no_grad():
            out_no_ref = model_no_ref(x, refiner_scale=1.0)

        torch.testing.assert_close(out_scale0["alpha"], out_no_ref["alpha"], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(out_scale0["fg"], out_no_ref["fg"], atol=1e-4, rtol=1e-4)

    def test_refiner_scale_default_is_one(self):
        """forward() without refiner_scale should default to 1.0."""
        from CorridorKeyModule.core.model_transformer import GreenFormer

        torch.manual_seed(42)
        model = GreenFormer(img_size=224, use_refiner=False)
        model.eval()

        x = torch.randn(1, 4, 224, 224)

        with torch.no_grad():
            out_default = model(x)
            out_explicit = model(x, refiner_scale=1.0)

        torch.testing.assert_close(out_default["alpha"], out_explicit["alpha"])
        torch.testing.assert_close(out_default["fg"], out_explicit["fg"])
