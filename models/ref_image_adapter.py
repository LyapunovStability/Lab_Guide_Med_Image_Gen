from typing import Optional, Sequence

import torch
import torch.nn as nn
from diffusers import T2IAdapter


class RefImageAdapter(nn.Module):
    """
    Wrap a diffusers T2I-Adapter so `ref_img` can be injected into the UNet
    via `down_intrablock_additional_residuals`.
    """

    def __init__(
        self,
        adapter_model_id: Optional[str] = None,
        adapter_pretrained_path: Optional[str] = None,
        in_channels: int = 3,
        channels: Optional[Sequence[int]] = None,
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        adapter_type: str = "full_adapter",
        conditioning_scale: float = 1.0,
    ):
        super().__init__()

        if channels is None:
            channels = (320, 640, 1280, 1280)
        else:
            channels = tuple(channels)

        if conditioning_scale <= 0:
            raise ValueError("conditioning_scale must be positive.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive.")
        if downscale_factor <= 0:
            raise ValueError("downscale_factor must be positive.")
        if adapter_model_id and adapter_pretrained_path:
            raise ValueError("Provide only one of adapter_model_id or adapter_pretrained_path.")

        self.adapter_source = adapter_pretrained_path or adapter_model_id
        self.conditioning_scale = conditioning_scale
        self.adapter = self._build_adapter(
            adapter_source=self.adapter_source,
            in_channels=in_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            downscale_factor=downscale_factor,
            adapter_type=adapter_type,
        )

    def _build_adapter(
        self,
        adapter_source: Optional[str],
        in_channels: int,
        channels: Sequence[int],
        num_res_blocks: int,
        downscale_factor: int,
        adapter_type: str,
    ) -> T2IAdapter:
        if adapter_source:
            return T2IAdapter.from_pretrained(adapter_source)

        return T2IAdapter(
            in_channels=in_channels,
            channels=list(channels),
            num_res_blocks=num_res_blocks,
            downscale_factor=downscale_factor,
            adapter_type=adapter_type,
        )

    def forward(
        self,
        ref_img: torch.Tensor,
        conditioning_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
    ):
        scale = self.conditioning_scale if conditioning_scale is None else conditioning_scale
        if scale <= 0:
            raise ValueError("conditioning_scale must be positive.")
        if num_images_per_prompt <= 0:
            raise ValueError("num_images_per_prompt must be positive.")
        if ref_img.dim() != 4:
            raise ValueError("ref_img must be a 4D tensor of shape (B, C, H, W).")

        adapter_state = self.adapter(ref_img)
        adapter_state = [state * scale for state in adapter_state]

        if num_images_per_prompt > 1:
            adapter_state = [state.repeat(num_images_per_prompt, 1, 1, 1) for state in adapter_state]

        if do_classifier_free_guidance:
            adapter_state = [torch.cat([state] * 2, dim=0) for state in adapter_state]

        # Clone to match the official adapter pipeline pattern and avoid
        # accidental in-place mutation inside some diffusers versions.
        return [state.clone() for state in adapter_state]
