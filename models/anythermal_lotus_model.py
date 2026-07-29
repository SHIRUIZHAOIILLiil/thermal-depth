"""Minimal AnyThermal -> Adapter -> Lotus-G wrapper."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.anythermal_encoder import AnyThermalEncoder, ImageLike
from models.anythermal_lotus_adapter import AnyThermalLotusAdapter


@dataclass(frozen=True)
class AnyThermalFeatureInfo:
    hidden_state_indices: Tuple[int, ...]
    transformer_block_indices: Tuple[int, ...]
    grid_size: Tuple[int, int]
    preprocessed_shape: Tuple[int, ...]
    original_shape: Tuple[int, ...]
    num_register_tokens: int
    has_cls_token: bool


def _module_device(module: nn.Module, fallback: Optional[torch.device] = None) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    if fallback is None:
        raise RuntimeError(f"Cannot infer device from parameterless {type(module).__name__}")
    return fallback


def _module_dtype(module: nn.Module, fallback: Optional[torch.dtype] = None) -> torch.dtype:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.dtype
    if fallback is None:
        raise RuntimeError(f"Cannot infer dtype from parameterless {type(module).__name__}")
    return fallback


def _resolve_block_indices(
    requested: Optional[Sequence[int]],
    hidden_state_count: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if hidden_state_count < 2:
        raise RuntimeError(
            "AnyThermal output_hidden_states must include embedding plus block outputs."
        )
    block_count = hidden_state_count - 1
    requested = tuple(requested) if requested is not None else (-4, -3, -2, -1)
    if len(requested) != 4:
        raise ValueError("Adapter V0 requires exactly four transformer block indices.")

    block_indices = []
    hidden_indices = []
    for index in requested:
        block_index = block_count + index if index < 0 else index
        if block_index < 0 or block_index >= block_count:
            raise ValueError(
                f"Requested transformer block {index} resolves to {block_index}, "
                f"but valid block indices are 0..{block_count - 1}."
            )
        block_indices.append(block_index)
        hidden_indices.append(block_index + 1)
    return tuple(hidden_indices), tuple(block_indices)


def _tokens_to_spatial(
    tokens: torch.Tensor,
    *,
    preprocessed_shape: Tuple[int, ...],
    patch_size: int,
    num_register_tokens: int,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise RuntimeError(f"Expected hidden state tokens [B,N,C], got {tuple(tokens.shape)}.")
    patch_start = 1 + num_register_tokens
    patch_tokens = tokens[:, patch_start:]
    grid_height = preprocessed_shape[-2] // patch_size
    grid_width = preprocessed_shape[-1] // patch_size
    expected = grid_height * grid_width
    if patch_tokens.shape[1] != expected:
        raise RuntimeError(
            "Cannot restore AnyThermal spatial features: "
            f"{patch_tokens.shape[1]} patch tokens vs grid {grid_height}x{grid_width}."
        )
    batch_size, _, channels = patch_tokens.shape
    return (
        patch_tokens.reshape(batch_size, grid_height, grid_width, channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def extract_anythermal_feature_pyramid(
    encoder: AnyThermalEncoder,
    thermal_image: ImageLike,
    *,
    feature_layer_indices: Optional[Sequence[int]] = None,
    enable_grad: bool = False,
) -> Tuple[Sequence[torch.Tensor], AnyThermalFeatureInfo, Dict[str, Any]]:
    """Extract four frozen AnyThermal block features as `[B,768,H,W]` maps."""
    inputs, original_shape = encoder.preprocess(thermal_image)
    grad_context = nullcontext() if enable_grad else torch.no_grad()
    with grad_context:
        try:
            outputs = encoder.model(**inputs, output_hidden_states=True)
        except TypeError as exc:
            raise RuntimeError(
                "AnyThermal model does not accept output_hidden_states=True; "
                "cannot build the four-layer feature pyramid."
            ) from exc

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("AnyThermal output does not contain hidden_states.")

    hidden_indices, block_indices = _resolve_block_indices(
        feature_layer_indices,
        len(hidden_states),
    )
    preprocessed_shape = tuple(inputs["pixel_values"].shape)
    features = [
        _tokens_to_spatial(
            hidden_states[index],
            preprocessed_shape=preprocessed_shape,
            patch_size=encoder.patch_size,
            num_register_tokens=encoder.num_register_tokens,
        )
        for index in hidden_indices
    ]
    info = AnyThermalFeatureInfo(
        hidden_state_indices=hidden_indices,
        transformer_block_indices=block_indices,
        grid_size=(features[0].shape[-2], features[0].shape[-1]),
        preprocessed_shape=preprocessed_shape,
        original_shape=tuple(original_shape),
        num_register_tokens=encoder.num_register_tokens,
        has_cls_token=True,
    )
    return features, info, encoder._require_thermal_diagnostics()


def latent_valid_mask(
    valid_mask: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Downsample pixel-space validity to the VAE latent grid like Lotus training."""
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
        raise ValueError(
            "valid_mask must have shape [B,1,H,W] or [B,H,W], "
            f"got {tuple(valid_mask.shape)}."
        )
    valid_mask = valid_mask.to(device=target_latents.device).bool()
    pooled_valid = F.max_pool2d(
        valid_mask.float(),
        kernel_size=vae_scale_factor,
        stride=vae_scale_factor,
    ).bool()
    if pooled_valid.shape[-2:] != target_latents.shape[-2:]:
        pooled_valid = F.interpolate(
            pooled_valid.float(),
            size=target_latents.shape[-2:],
            mode="nearest",
        ).bool()
    return pooled_valid.repeat(1, target_latents.shape[1], 1, 1)


class AnyThermalLotusModel(nn.Module):
    """Small training wrapper for adapter-only Lotus-G experiments."""

    def __init__(
        self,
        *,
        anythermal_encoder: AnyThermalEncoder,
        lotus_pipeline: Any,
        adapter: Optional[nn.Module] = None,
        noise_scheduler: Optional[Any] = None,
        feature_layer_indices: Optional[Sequence[int]] = None,
        freeze_anythermal: bool = True,
        freeze_lotus: bool = True,
    ) -> None:
        super().__init__()
        self.anythermal_encoder = anythermal_encoder
        self.lotus = lotus_pipeline
        self.adapter = adapter or AnyThermalLotusAdapter()
        self.noise_scheduler = noise_scheduler or lotus_pipeline.scheduler
        self.feature_layer_indices = (
            tuple(feature_layer_indices) if feature_layer_indices is not None else None
        )

        if freeze_anythermal:
            self.anythermal_encoder.model.eval()
            self.anythermal_encoder.model.requires_grad_(False)
        if freeze_lotus:
            for module_name in ("vae", "text_encoder", "unet"):
                module = getattr(self.lotus, module_name, None)
                if module is not None:
                    module.eval()
                    module.requires_grad_(False)
        self.adapter.train()

    @property
    def device(self) -> torch.device:
        return _module_device(self.adapter, _module_device(self.lotus.unet))

    def trainable_parameters(self) -> Sequence[nn.Parameter]:
        return [parameter for parameter in self.adapter.parameters() if parameter.requires_grad]

    def extract_features(
        self,
        thermal_image: ImageLike,
    ) -> Tuple[Sequence[torch.Tensor], AnyThermalFeatureInfo, Dict[str, Any]]:
        return extract_anythermal_feature_pyramid(
            self.anythermal_encoder,
            thermal_image,
            feature_layer_indices=self.feature_layer_indices,
            enable_grad=False,
        )

    def encode_depth_latents(self, depth_values: torch.Tensor) -> torch.Tensor:
        if depth_values.ndim != 4 or depth_values.shape[1] != 3:
            raise ValueError(
                "depth_values must have shape [B,3,H,W] in Lotus VAE range [-1,1], "
                f"got {tuple(depth_values.shape)}."
            )
        vae = self.lotus.vae
        device = _module_device(vae)
        dtype = _module_dtype(vae)
        with torch.no_grad():
            latents = vae.encode(depth_values.to(device=device, dtype=dtype)).latent_dist.sample()
            return latents * vae.config.scaling_factor

    def encode_empty_prompt(self, batch_size: int) -> torch.Tensor:
        text_encoder = self.lotus.text_encoder
        tokenizer = self.lotus.tokenizer
        device = _module_device(text_encoder)
        with torch.no_grad():
            text_inputs = tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            return text_encoder(input_ids, return_dict=False)[0]

    def depth_task_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        task_emb = torch.tensor([[1.0, 0.0]], device=device).repeat(batch_size, 1)
        return torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    def forward(
        self,
        *,
        thermal_image: Optional[ImageLike] = None,
        features: Optional[Sequence[torch.Tensor]] = None,
        depth_values: torch.Tensor,
        timesteps: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        return_decoded: bool = True,
    ) -> Dict[str, Any]:
        if features is None:
            if thermal_image is None:
                raise ValueError("Either thermal_image or precomputed features must be provided.")
            features, feature_info, thermal_diagnostics = self.extract_features(thermal_image)
        else:
            feature_info = None
            thermal_diagnostics = None

        target_latents = self.encode_depth_latents(depth_values)
        target_size = tuple(target_latents.shape[-2:])
        feature_device = features[-1].device
        feature_dtype = features[-1].dtype
        adapter_device = _module_device(self.adapter, feature_device)
        adapter_dtype = _module_dtype(self.adapter, feature_dtype)
        features = [
            feature.to(device=adapter_device, dtype=adapter_dtype)
            for feature in features
        ]
        condition_latent = self.adapter(features, target_size=target_size)

        batch_size = condition_latent.shape[0]
        unet = self.lotus.unet
        unet_device = _module_device(unet)
        unet_dtype = _module_dtype(unet)
        target_latents = target_latents.to(device=unet_device, dtype=unet_dtype)
        condition_for_unet = condition_latent.to(device=unet_device, dtype=unet_dtype)
        timesteps = timesteps.to(device=unet_device, dtype=torch.long).view(-1)
        if timesteps.numel() == 1 and batch_size > 1:
            timesteps = timesteps.repeat(batch_size)
        if timesteps.numel() != batch_size:
            raise ValueError(
                f"timesteps must have one value or batch_size={batch_size} values, "
                f"got {timesteps.numel()}."
            )

        if noise is None:
            noise = torch.randn_like(target_latents)
        else:
            noise = noise.to(device=unet_device, dtype=unet_dtype)
        noisy_depth_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        unet_input = torch.cat([condition_for_unet, noisy_depth_latents], dim=1)

        if prompt_embeds is None:
            prompt_embeds = self.encode_empty_prompt(batch_size)
        prompt_embeds = prompt_embeds.to(device=unet_device, dtype=unet_dtype)
        task_emb = self.depth_task_embedding(batch_size, unet_device).to(dtype=unet_dtype)

        model_pred = unet(
            unet_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            class_labels=task_emb,
        )[0]

        if valid_mask is None:
            loss_mask = torch.ones_like(target_latents, dtype=torch.bool)
        else:
            loss_mask = latent_valid_mask(
                valid_mask,
                target_latents,
                vae_scale_factor=int(getattr(self.lotus, "vae_scale_factor", 8)),
            )
        if not bool(loss_mask.any()):
            raise RuntimeError("No valid latent pixels are available for loss computation.")
        loss = F.mse_loss(
            model_pred.float()[loss_mask],
            target_latents.float()[loss_mask],
            reduction="mean",
        )

        decoded = None
        if return_decoded:
            with torch.no_grad():
                decoded = self.lotus.vae.decode(
                    model_pred.detach() / self.lotus.vae.config.scaling_factor,
                    return_dict=False,
                )[0]

        return {
            "loss": loss,
            "condition_latent": condition_latent,
            "target_latents": target_latents,
            "noisy_depth_latents": noisy_depth_latents,
            "unet_input": unet_input,
            "model_pred": model_pred,
            "decoded": decoded,
            "prompt_embeds": prompt_embeds,
            "task_emb": task_emb,
            "loss_mask": loss_mask,
            "feature_info": feature_info,
            "thermal_diagnostics": thermal_diagnostics,
        }

